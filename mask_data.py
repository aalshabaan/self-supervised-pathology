# Applies attention masks to the first hundred images of each class of the dataset.
import functools
import os
import sys
from platform import node
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.io import imsave
from matplotlib import colormaps
from tqdm import tqdm

sys.path.extend([os.getcwd(), os.path.join(os.getcwd(), 'facebookresearch_dino_main')])

from facebookresearch_dino_main.visualize_attention import apply_mask
from facebookresearch_dino_main.vision_transformer import VisionTransformer
from Abed_utils import get_model, get_data_loader, OUTPUT_ROOT, output_paths

def mask_batch(model:VisionTransformer, batch, names, labels, im_size, patch_size, threshold=0.5, device='cuda', relative_mask=True):

    global output_paths

    im_size = (im_size, im_size) if isinstance(im_size, int) else im_size

    w_featmap = im_size[0] // patch_size
    h_featmap = im_size[1] // patch_size

    attn = model.get_last_selfattention(batch.to(device))
    nh = attn.shape[1]
    attn = attn[:, :, 0, 1:].reshape(attn.shape[0], nh, -1)


    if relative_mask:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attn)
        val /= torch.sum(val, dim=2, keepdim=True)
        cumval = torch.cumsum(val, dim=2)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for i in range(idx2.shape[0]):
            for head in range(nh):
                th_attn[i,head] = th_attn[i,head][idx2[i,head]]
    else:
        th_attn = torch.empty(*attn.shape)
        for i in range(attn.shape[0]):
            for head in range(nh):
                th_attn[i,head] = attn[i,head] > threshold*(attn[i,head].min() + attn[i,head].max())
    th_attn = th_attn.reshape(th_attn.shape[0], nh, w_featmap, h_featmap).float()
    # interpolate
    th_attn = F.interpolate(th_attn, scale_factor=patch_size, mode="nearest").bool().detach().cpu().numpy()


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
    patch_size = 8
    im_size = 224
    model = get_model(patch_size, './ckpts/checkpoint0018.pth', key='teacher', device=device)
    print('Loading the dataset')
    data = get_data_loader(im_size, patch_size, 10, whole_slide=False, output_subdir='dino_finetuning_abs')
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for i, (batch, fname, lbl) in enumerate(tqdm(data)):
        names = [os.path.basename(x).split(os.path.extsep)[0] for x in list(fname)]
        mask_batch(model, batch, names, lbl, im_size, patch_size, device=device, threshold=0.5, relative_mask=False)

        if i>9:
            break