import os
import random

import sentencepiece
import torch

DICT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datamodule",
    "char_units.txt"
)

class TextTransform:
    """Mapping Dictionary Class for SentencePiece tokenization."""

    def __init__(
        self,
        dict_path=DICT_PATH,
    ):
        # Load units and create dictionary
        units = open(dict_path, encoding='utf8').read().splitlines()
        self.hashmap = {unit.split()[0]: unit.split()[-1] for unit in units}
        # 0 will be used for "blank" in CTC
        self.token_list = ["<blank>"] + list(self.hashmap.keys()) + ["<eos>"]
        self.ignore_id = -1

    def tokenize(self, text):
        token_ids = [self.hashmap.get(token, self.hashmap["<unk>"]) for token in text]
        return torch.tensor(list(map(int, token_ids)))

    def post_process(self, token_ids):
        valid_ids = []
        for token in token_ids:
            if token != -1:
                valid_ids.append(token)
        token_ids = valid_ids
        text = self._ids_to_str(token_ids, self.token_list)
        text = text.replace("\u2581", " ").strip()
        return text

    def _ids_to_str(self, token_ids, char_list):
        token_as_list = [char_list[idx] for idx in token_ids]
        return "".join(token_as_list).replace("<space>", " ")