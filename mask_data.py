# Applies attention masks to the first hundred images of each class of the dataset.
# TODO Check masked output dtype and convert properly
import functools
import os
import sys

from platform import node

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from skimage.io import imread, imsave
from matplotlib import colormaps
from tqdm import tqdm

from facebookresearch_dino_main.visualize_attention import apply_mask
from facebookresearch_dino_main.vision_transformer import vit_small, VisionTransformer

PC_NAME = node()

DATA_ROOT = 'D:/self_supervised_pathology/datasets/NCT-CRC-HE-100K-NONORM'\
    if 'Abdulrahman' in PC_NAME else '/mnt/data/dataset/tiled/kather19tiles_nonorm/NCT-CRC-HE-100K-NONORM'

OUTPUT_ROOT = 'D:/self_supervised_pathology/output/attention/dino_tcga_brca/NCT-CRC-HE-100K-NONORM'\
    if 'Abdulrahman' in PC_NAME else '/home/guest/Documents/shabaan_2022/output/attention/dino_tcga_brca/NCT-CRC-HE-100K-NONORM'

output_paths = []

sys.path.extend([os.getcwd(), os.path.join(os.getcwd(), 'facebookresearch_dino_main')])

_model = None

class NamedImageFolder(ImageFolder):
    def __getitem__(self, item):
        return self.transform(self.loader(self.imgs[item][0])), *self.imgs[item]

def get_model(patch_size, pretrained_weight_path, device='cuda'):
    if patch_size not in [8, 16]:
        raise ValueError('patch size must be 8 or 16')
    global _model
    if _model is None:
        _model = vit_small(patch_size=patch_size, num_classes=0)
        state_dict = torch.load(pretrained_weight_path)
        if 'tcga' in pretrained_weight_path:
            state_dict = state_dict['teacher']
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

def get_data_loader(im_size, patch_size, batch_size=1):
    global output_paths
    t = functools.partial(normalize_input, im_size=im_size, patch_size=patch_size)
    ds = NamedImageFolder(DATA_ROOT, transform=t, loader=imread)
    for i,cls in enumerate(ds.classes):
        output_paths.append(os.path.join(OUTPUT_ROOT, cls))
        os.makedirs(output_paths[i], exist_ok=True)
    dl = DataLoader(ds, batch_size, shuffle=True)
    return dl

def process_batch(model:VisionTransformer, batch, names, labels, im_size, patch_size, threshold=None, device='cuda'):

    global output_paths

    im_size = (im_size, im_size) if isinstance(im_size, int) else im_size

    w_featmap = im_size[0] // patch_size
    h_featmap = im_size[1] // patch_size

    attn = model.get_last_selfattention(batch.to(device))
    nh = attn.shape[1]
    attn = attn[:, :, 0, 1:].reshape(attn.shape[0], nh, -1)

    if threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attn)
        val /= torch.sum(val, dim=2, keepdim=True)
        cumval = torch.cumsum(val, dim=2)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for i in range(idx2.shape[0]):
            for head in range(nh):
                th_attn[i,head] = th_attn[i,head][idx2[i,head]]
        th_attn = th_attn.reshape(th_attn.shape[0], nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn, scale_factor=patch_size, mode="nearest").bool().detach().cpu().numpy()

    colors = colormaps['tab10'].colors[:nh]

    for i in range(th_attn.shape[0]):
        x = np.moveaxis(batch[i].detach().numpy(), 0, 2)
        x = (255 * (x * [0.229, 0.224, 0.225] + (0.485, 0.456, 0.406))).astype(np.uint8)
        outpath = os.path.join(output_paths[labels[i]], names[i])
        os.makedirs(outpath, exist_ok=True)
        imsave(os.path.join(outpath, 'input.png'), x)
        for head in range(th_attn.shape[1]):
            out = apply_mask(x, th_attn[i,head], colors[head])
            imsave(os.path.join(outpath, f'mask_head{head}.png'), out)



if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    patch_size = 16
    im_size = 224
    model = get_model(patch_size, './ckpts/vits_tcga_brca_dino.pt', device)
    print('Loading the dataset')
    data = get_data_loader(im_size, patch_size, 10)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for i, (batch, fname, lbl) in tqdm(enumerate(data)):
        names = [os.path.basename(x).split(os.path.extsep)[0] for x in list(fname)]
        process_batch(model, batch, names, lbl, im_size, patch_size, device=device, threshold=0.5)

        if i>9:
            break