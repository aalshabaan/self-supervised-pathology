import functools
import os

import Abed_utils
from wsi import WholeSlideDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm



if __name__ == '__main__':
    patch_size = 8
    batch_size = 8
    im_size = 224
    K = 20
    t = functools.partial(Abed_utils.normalize_input, im_size=im_size, patch_size=patch_size)
    ds = WholeSlideDataset(Abed_utils.WHOLE_SLIDE_PATH, transform=t)
    data = DataLoader(ds, batch_size=batch_size)

    cmap = plt.colormaps['tab10'].colors

    print('Loading Kather-19 embeddings')
    features, labels = Abed_utils.load_features(os.path.join(Abed_utils.OUTPUT_ROOT, 'features'), load_labels=True, cuda=True)

    print('Creating model and loading pretrained weights')
    classifier = Abed_utils.KNN_classifier(features, labels, K) # uses cuda if available

    backbone = Abed_utils.get_model(patch_size, './ckpts/dino_deitsmall8_pretrain.pth')
    backbone.eval()

    model = nn.Sequential(backbone, classifier)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    canvas = torch.ones(*ds.s.dimensions, dtype=torch.uint8)*255
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

        if not (i % 100):
            print('saving')
            torch.save(canvas, os.path.join(outpath, 'ckpt.pt'))

    torch.save(canvas, os.path.join(outpath, 'seg_dino_imagenet_100ep.pt'))
    print('Done calculating, convert to image using black magic to not run out of memory')


