import functools
import os
import sys
from platform import node

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from skimage.io import imread

sys.path.extend([os.getcwd(), os.path.join(os.getcwd(), 'facebookresearch_dino_main')])

from facebookresearch_dino_main.vision_transformer import vit_small
from wsi import WholeSlideDataset

PC_NAME = node()

DATA_ROOT = 'D:/self_supervised_pathology/datasets/NCT-CRC-HE-100K-NONORM'\
    if 'Abdulrahman' in PC_NAME else '/mnt/data/dataset/tiled/kather19tiles_nonorm/NCT-CRC-HE-100K-NONORM'

OUTPUT_ROOT = 'D:/self_supervised_pathology/output/'\
    if 'Abdulrahman' in PC_NAME else '/home/guest/Documents/shabaan_2022/output/'

WHOLE_SLIDE_PATH = "D:/self_supervised_pathology/datasets/WSI/TCGA-CK-6747-01Z-00-DX1.7824596c-84db-4bee-b149-cd8f617c285f.svs"\
    if 'Abdulrahman' in PC_NAME else None

output_paths = []



_model = None
class NamedImageFolder(ImageFolder):
    def __getitem__(self, item):
        return self.transform(self.loader(self.imgs[item][0])), *self.imgs[item]

def get_model(patch_size, pretrained_weight_path, key=None, device='cuda'):
    if patch_size not in [8, 16]:
        raise ValueError('patch size must be 8 or 16')
    global _model
    if _model is None:
        _model = vit_small(patch_size=patch_size, num_classes=0)
        state_dict = torch.load(pretrained_weight_path)
        if key is not None:
            state_dict = state_dict[key]
            to_remove = []
            for key in state_dict.keys():
                if 'head' in key:
                    to_remove.append(key)
            for key in to_remove:
                state_dict.pop(key)
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = _model.load_state_dict(state_dict)
        print(f'state dict loaded with the message {msg}')
        _model = _model.to(device)
    return _model

def normalize_input(input, im_size, patch_size):
    # print(type(input), input)
    t = transforms.Compose([transforms.ToTensor(), transforms.Resize(im_size), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    img = t(input)

    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h]

    return img

def get_data_loader(im_size, patch_size, batch_size=1, whole_slide=False, output_subdir=None):
    global output_paths
    global OUTPUT_ROOT
    output_subdir = os.path.join(OUTPUT_ROOT, output_subdir) if output_subdir is not None else OUTPUT_ROOT
    t = functools.partial(normalize_input, im_size=im_size, patch_size=patch_size)
    if whole_slide:
        if WHOLE_SLIDE_PATH is None:
            raise ValueError('Whole slide image is not available on this machine')
        ds = WholeSlideDataset(WHOLE_SLIDE_PATH, transform=t, )
        slide_name = os.path.split(WHOLE_SLIDE_PATH)[1].split(os.extsep)[0]
        output_paths.append(os.path.join(output_subdir, slide_name))
    else:
        ds = NamedImageFolder(DATA_ROOT, transform=t, loader=imread)
        for i,cls in enumerate(ds.classes):
            output_paths.append(os.path.join(output_subdir, cls))
            os.makedirs(output_paths[i], exist_ok=True)
    dl = DataLoader(ds, batch_size, shuffle=True)
    return dl