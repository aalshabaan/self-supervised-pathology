# Applies attention masks to the first hundred images of each class of the dataset.
# TODO Check masked output dtype and convert properly
import functools
import os
import sys

from platform import node

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from skimage.io import imread, imsave
from matplotlib import colormaps
from tqdm import tqdm

from facebookresearch_dino_main.visualize_attention import apply_mask, random_colors
from facebookresearch_dino_main.vision_transformer import vit_small, VisionTransformer

PC_NAME = node()

DATA_ROOT = 'D:/self_supervised_pathology/output/attention/dino_base/NCT-CRC-HE-100K-NONORM'\
    if 'Abdulrahman' in PC_NAME else '/mnt/data/dataset/tiled/kather19tiles_nonorm/NCT-CRC-HE-100K-NONORM'

OUTPUT_FOLDER = 'D:/self_supervised_pathology/output/attention/dino_base/NCT-CRC-HE-100K-NONORM'\
    if 'Abdulrahman' in PC_NAME else '/home/guest/Documents/shabaan_2022/output/attention/dino_base/NCT-CRC-HE-100K-NONORM'

sys.path.extend([os.getcwd(), os.path.join(os.getcwd(), 'facebookresearch_dino_main')])

_model = None
def get_model(patch_size, pretrained_weight_path, device='cuda'):
    if patch_size not in [8, 16]:
        raise ValueError('patch size must be 8 or 16')
    global _model
    if _model is None:
        _model = vit_small(patch_size=patch_size, num_classes=0)
        state_dict = torch.load(pretrained_weight_path)
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
    t = transforms.Compose([transforms.Resize(im_size), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    img = t(input)

    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h]

    return img

def get_data_loader(im_size, patch_size, batch_size=1):
    t = functools.partial(normalize_input, im_size=im_size, patch_size=patch_size)
    ds = ImageFolder(DATA_ROOT, transform=t)
    dl = DataLoader(ds, batch_size, shuffle=False)
    return dl

def process_batch(model:VisionTransformer, batch, im_size, patch_size, threshold=None, device='cuda'):
    # x = normalize_input(batch, im_size, patch_size)

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
        for head in range(nh):
            th_attn[:,head] = th_attn[:,head,idx2[:,head]]
        th_attn = th_attn.reshape(th_attn.shape[0], nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn, scale_factor=patch_size, mode="nearest").bool().detach().cpu().numpy()

    colors = random_colors(nh)
    for i in range(th_attn.shape[0]):
        for head in range(th_attn.shape[1]):
            x = np.moveaxis(255*batch[i].detach().numpy().astype(np.uint8), 0, 2)
            # print(i,head,th_attn[i,head].min(), th_attn[i,head].max())
            out = apply_mask(x, th_attn[i,head], colors[head])
            # print(out.min(), out.max())
            # break
        imsave(os.path.join(OUTPUT_FOLDER, f'out{i}.png'), out)




if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    patch_size = 8
    im_size = 224
    model = get_model(patch_size, './ckpts/dino_deitsmall8_300ep_pretrain.pth', device)
    print('Loading the dataset')
    data = get_data_loader(im_size, patch_size, 1)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for batch, lbl in tqdm(data):
        process_batch(model, batch, im_size, patch_size, device=device, threshold=0.5)
        break
