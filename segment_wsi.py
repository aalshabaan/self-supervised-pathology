import functools
import math
import os

import Abed_utils
from wsi import WholeSlideDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import typing


def wsi_tensor_into_image(path_to_tensor:typing.Union[str, torch.Tensor], downsampling=10, cmap='tab10', n_classes=9, class_labels=None):

    # clone to free the original huge tensor for garbage collection
    img = torch.load(path_to_tensor)[::downsampling, ::downsampling].clone().t().unsqueeze(2).repeat(1, 1, 3).numpy()
    # print(img.shape)
    if cmap in plt.colormaps:
        colors = plt.colormaps[cmap].colors
    else:
        print(f'invalid colormap {cmap}, using tab10 instead')
        colors = plt.colormaps['tab10'].colors

    colors = list(colors)
    for i in range(n_classes):
        colors[i] = list(colors[i])
        for j in range(3):
            colors[i][j] = round(colors[i][j]*255)
        colors[i] = tuple(colors[i])

    for i in range(n_classes):
        for c in range(3):
            # print(img.shape, colors.shape)
            img[:, :, c][img[:, :, c] == i] = colors[i][c]

    if class_labels is not None:
        legend = Image.new('RGB', (100, img.shape[0]), (255, 255, 255))
        draw = ImageDraw.ImageDraw(legend)
        font = ImageFont.truetype('arial.ttf', 15)
        for i in range(n_classes):
            if i < len(class_labels):
                lbl = class_labels[i]
            else:
                lbl = str(i)
            # box = Image.new('RGB', (math.ceil(legend.width/2), legend.height//9), colors[i])
            draw.rectangle((legend.width/2, i*legend.height/n_classes, legend.width, (i+1)*legend.height/n_classes), fill=colors[i])
            # legend.paste(box, (i*box.height, legend.width/2))
            draw.text((0, (i+0.5)*legend.height/n_classes), text=lbl, font=font, fill=(0,0,0))
        # img:np.ndarray
        img = np.pad(img, ((0, 0), (0, legend.width), (0, 0)))
        img = Image.fromarray(img)
        img.paste(legend, (img.width-legend.width, 0))
        img = np.array(img)

    outpath, filename = os.path.split(path_to_tensor)
    filename = filename.split(os.extsep)[0]+'.png'
    plt.imsave(os.path.join(outpath, filename), img)

def segment_wsi(path_to_wsi, path_to_model, path_to_embeddings, out_filename, K=20, wsi_patch_size=224,
                vit_patch_size=8, batch_size=8):
    t = functools.partial(Abed_utils.normalize_input, im_size=wsi_patch_size, patch_size=vit_patch_size)
    ds = WholeSlideDataset(path_to_wsi, transform=t, crop_sizes_px=[wsi_patch_size])
    data = DataLoader(ds, batch_size=batch_size)

    print('Loading Kather-19 embeddings')
    features, labels = Abed_utils.load_features(path_to_embeddings, load_labels=True, cuda=True)

    print('Creating model and loading pretrained weights')
    classifier = Abed_utils.KNN_classifier(features, labels, K)  # uses the device of the feature/label tensors

    backbone = Abed_utils.get_model(vit_patch_size, path_to_model)
    backbone.eval()

    model = nn.Sequential(backbone, classifier)

    canvas = torch.ones(*ds.s.dimensions, dtype=torch.uint8) * 255
    print(f'Created canvas of size {canvas.size()}')

    outpath = os.path.join(Abed_utils.OUTPUT_ROOT, 'wsi', os.path.split(Abed_utils.WHOLE_SLIDE_PATH)[1])
    os.makedirs(outpath, exist_ok=True)

    for i, crop_pair in enumerate(tqdm(data)):
        for img, metas in zip(*crop_pair):
            tx, ty, bx, by = metas[2].long(), metas[3].long(), metas[6].long(), metas[7].long()
            pred = model(img.to(next(model.parameters()).device))
            # print(tx.shape, ty.shape, bx.shape, by.shape, pred.shape)
            for i in range(pred.shape[0]):
                canvas[tx[i]:bx[i], ty[i]:by[i]] = pred[i]

        if i % 100 == 0:
            print('saving')
            torch.save(canvas, os.path.join(outpath, 'ckpt.pt'))

    torch.save(canvas, os.path.join(outpath, out_filename))
    return  outpath

if __name__ == '__main__':

    feat_path = os.path.join(Abed_utils.OUTPUT_ROOT, 'features')
    out_filename = 'seg_dino_imagenet_100ep_ds.pt'

    outpath = segment_wsi(Abed_utils.WHOLE_SLIDE_PATH, './ckpts/dino_deitsmall8_pretrain.pth', feat_path, out_filename)

    print('Outputting image')
    wsi_tensor_into_image(os.path.join(outpath, out_filename))



