import functools
import os.path
import time
from typing import Iterable
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as tr

from Abed_utils import get_data_loader, get_model, OUTPUT_ROOT, K_19_PATH, normalize_input, load_tif_windows
from facebookresearch_dino_main.eval_knn import ReturnIndexDataset
import facebookresearch_dino_main.utils as utils


@torch.no_grad()
def save_features(model:torch.nn.Module, data_loader:DataLoader, out_dir, multiscale=False):
    print('Extracting features')
    # metric_logger = utils.MetricLogger(delimiter="  ")
    out_dim = None
    for m in model.modules():
        if hasattr(m, 'out_features'):
            out_dim = m.out_features
    if out_dim is None:
        raise RuntimeError('Model output dimensionality can not be inferred')

    features = torch.zeros(len(data_loader.dataset), out_dim).cpu()
    print(f'Saving features in tensor of shape {features.shape}')
    time.sleep(0.01)
    for samples, index in tqdm(data_loader, desc='Feature extraction...'):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        labels = torch.tensor([s[-1] for s in data_loader.dataset.samples]).long()
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone()
        features[index] = feats.cpu()

    # features = extract_features(model, data)
    print(f'saving to {out_dir}')
    os.makedirs(out_dir, exist_ok=True)
    torch.save(features, os.path.join(out_dir, 'features.pt'))
    torch.save(labels, os.path.join(out_dir, 'labels.pt'))

if __name__ == '__main__':
    # data = get_data_loader(224, 16, 64, dataset_class=ReturnIndexDataset, shuffle=False)
    t = functools.partial(normalize_input, im_size=224, patch_size=8)
    transofrm = tr.Compose([t,
                            tr.RandomHorizontalFlip(p=1)])
    ds = ReturnIndexDataset(K_19_PATH, transform=transofrm, loader=load_tif_windows)
    data = DataLoader(ds, 64)
    model = get_model(8, pretrained_weight_path='ckpts/dino_deitsmall8_pretrain.pth')
    save_features(model, data, out_dir=os.path.join(OUTPUT_ROOT, 'features_flipped'))
