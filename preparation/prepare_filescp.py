
import os
import random
import argparse

pwd_dir = os.path.dirname(os.path.abspath('.')) # preparation
filelist_dir = os.path.join(pwd_dir, 'data')

def prepare_cncvs_scp(cncvs_rootdir, dst):
    filelist = []
    for split in os.listdir(cncvs_rootdir):
        for spk in os.listdir(f"{cncvs_rootdir}/{split}"):
            src_dir = os.path.join(cncvs_rootdir, split, spk, 'video')
            dst_dir = os.path.join(dst, split, spk, 'video')
            landmark_dir = os.path.join(dst, 'facelandmark')
            for fn in os.listdir(src_dir):
                vfn = fn[:-4]
                filelist.append(f"{vfn}\t{src_dir}\t{dst_dir}\t{landmark_dir}")
    random.shuffle(filelist)
    with open('cncvs.scp', 'w') as fp:
        fp.write('\n'.join(filelist))
        
def prepare_ss_scp(single_speaker_rootdir, dst):
    filelist = []
    for csv in os.listdir(os.path.join(filelist_dir, 'single-speaker')):
        if csv.startswith('train'):
            continue
        for line in open(os.path.join(filelist_dir, 'single-speaker', csv)).readlines():
            _, path, _, _ = line.split(',')
            vfn = os.path.join(single_speaker_rootdir, os.path.dirname(path))
            dst_dir = os.path.join(dst, os.path.dirname(path))
            landmark_dir = os.path.join(dst, 'facelandmark')
            uid = os.path.basename(path).replace('.mp4', '')
            filelist.append(f"{uid}\t{vfn}\t{dst_dir}\t{landmark_dir}")
    random.shuffle(filelist)
    with open('single-speaker.scp', 'w') as fp:
        fp.write('\n'.join(filelist))


def prepare_ms_scp(multi_speaker_rootdir, dst):
    filelist = []
    for csv in os.listdir(os.path.join(filelist_dir, 'multi-speaker')):
        for line in open(os.path.join(filelist_dir, 'multi-speaker', csv)).readlines():
            _, path, _, _ = line.split(',')
            vfn = os.path.join(multi_speaker_rootdir, os.path.dirname(path))
            dst_dir = os.path.join(dst, os.path.dirname(path))
            landmark_dir = os.path.join(dst, 'facelandmark')
            uid = os.path.basename(path).replace('.mp4', '')
            filelist.append(f"{uid}\t{vfn}\t{dst_dir}\t{landmark_dir}")
    random.shuffle(filelist)
    with open('multi-speaker.scp', 'w') as fp:
        fp.write('\n'.join(filelist))
    
parser = argparse.ArgumentParser()
parser.add_argument('--src', required=True, help='source dir of downloaded files')
parser.add_argument('--dst', required=True, help='dst dir of processed video files')
if __name__== '__main__':
    args = parser.parse_args()
    src = args.src
    dst = args.dst
    prepare_cncvs_scp(f'{src}/cncvs/', f'{dst}/cncvs/')
    prepare_ss_scp(f'{src}/single-speaker/', f'{dst}/single-speaker/')
    prepare_ms_scp(f'{src}/multi-speaker/', f'{dst}/multi-speaker/')
