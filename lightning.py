from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torchaudio
from cosine import WarmupCosineScheduler
from datamodule.transforms import TextTransform

# for testing
from espnet.asr.asr_utils import add_results_to_json, get_model_conf, torch_load
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from pytorch_lightning import LightningModule


def compute_char_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1, seq2)

def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(
        seq1.lower().split(), seq2.lower().split()
    )


class ModelModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        if self.cfg.data.modality == "audio":
            self.backbone_args = self.cfg.model.audio_backbone
        elif self.cfg.data.modality == "video":
            self.backbone_args = self.cfg.model.visual_backbone

        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list
        self.model = E2E(len(self.token_list), self.backbone_args)

        # -- initialise
        if self.cfg.ckpt_path:
            ckpt = torch.load(
                self.cfg.ckpt_path, map_location=lambda storage, loc: storage
            )
            if self.cfg.transfer_frontend:
                tmp_ckpt = {
                    k: v
                    for k, v in ckpt["model_state_dict"].items()
                    if k.startswith("trunk.") or k.startswith("frontend3D.")
                }
                self.model.encoder.frontend.load_state_dict(tmp_ckpt)
            else:
                if self.cfg.remove_ctc:
                    dict_new = {}
                    for key, value in ckpt.items():
                        if key not in ['decoder.embed.0.weight', 'decoder.output_layer.weight', 'decoder.output_layer.bias',
                                       'r_decoder.embed.0.weight', 'r_decoder.output_layer.weight', 'r_decoder.output_layer.bias',
                                       'ctc.ctc_lo.weight', 'ctc.ctc_lo.bias']:
                            dict_new[key] = value
                    self.model.load_state_dict(dict_new, strict=False)
                else:
                    self.model.load_state_dict(ckpt)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "name": "model",
                    "params": self.model.parameters(),
                    "lr": self.cfg.optimizer.lr,
                }
            ],
            weight_decay=self.cfg.optimizer.weight_decay,
            betas=(0.9, 0.98),
        )
        scheduler = WarmupCosineScheduler(
            optimizer,
            self.cfg.optimizer.warmup_epochs,
            self.cfg.trainer.max_epochs,
            len(self.trainer.datamodule.train_dataloader()),
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def forward(self, sample):
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)
        enc_feat, _ = self.model.encoder(sample.unsqueeze(0).to(self.device), None)
        enc_feat = enc_feat.squeeze(0)
        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted = add_results_to_json(nbest_hyps, self.token_list)
        predicted = predicted.replace("▁", " ").strip().replace("<eos>", "")
        return predicted

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="val")

    def test_step(self, sample, sample_idx):
        enc_feat, _ = self.model.encoder(
            sample["input"].unsqueeze(0).to(self.device), None
        )
        enc_feat = enc_feat.squeeze(0)
        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted = add_results_to_json(nbest_hyps, self.token_list)
        predicted = predicted.replace("▁", " ").strip().replace("<eos>", "")

        token_id = sample["target"]
        actual = self.text_transform.post_process(token_id)

        self.total_edit_distance += compute_char_level_distance(actual, predicted)
        self.total_length += len(actual)
        if sample_idx % 100 == 1:
            print(f'{sample_idx + 1} utt, avgcer = {self.total_edit_distance / self.total_length}')
        return

    def _step(self, batch, batch_idx, step_type):
        loss, loss_ctc, loss_att, acc, loss_att_reversed, acc_reversed = self.model(
            batch["inputs"], batch["input_lengths"], batch["targets"]
        )
        batch_size = len(batch["inputs"])

        if step_type == "train":
            self.log("loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log(
                "loss_ctc",
                loss_ctc,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                "loss_att",
                loss_att,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                "decoder_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size
            )
            self.log(
                "loss_att_reversed",
                loss_att_reversed,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                "decoder_acc_reversed", acc_reversed, on_step=True, on_epoch=True, batch_size=batch_size
            )
        else:
            self.log("loss_val", loss, batch_size=batch_size)
            self.log("loss_ctc_val", loss_ctc, batch_size=batch_size)
            self.log("loss_att_val", loss_att, batch_size=batch_size)
            self.log("decoder_acc_val", acc, batch_size=batch_size)
            self.log("loss_att_reversed_val", loss_att_reversed, batch_size=batch_size)
            self.log("decoder_acc_reversed_val", acc_reversed, batch_size=batch_size)


        if step_type == "train":
            self.log(
                "monitoring_step", torch.tensor(self.global_step, dtype=torch.float32)
            )

        return loss

    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.loaders.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)
        return super().on_train_epoch_start()

    def on_test_epoch_start(self):
        self.total_length = 0
        self.total_edit_distance = 0
        self.text_transform = TextTransform()
        self.beam_search = get_beam_search_decoder(self.model, self.token_list, ctc_weight=0.5)

    def on_test_epoch_end(self):
        self.log("cer", self.total_edit_distance / self.total_length)


def get_beam_search_decoder(
    model,
    token_list,
    rnnlm=None,
    rnnlm_conf=None,
    penalty=0,
    ctc_weight=0.1,
    lm_weight=0.0,
    beam_size=40,
):
    sos = model.odim - 1
    eos = model.odim - 1
    scorers = model.scorers()

    if not rnnlm:
        lm = None
    else:
        lm_args = get_model_conf(rnnlm, rnnlm_conf)
        lm_model_module = getattr(lm_args, "model_module", "default")
        lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
        lm = lm_class(len(token_list), lm_args)
        torch_load(rnnlm, lm)
        lm.eval()

    scorers["lm"] = lm
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": lm_weight,
        "length_bonus": penalty,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )