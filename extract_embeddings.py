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

import Abed_utils
from Abed_utils import get_data_loader, get_vit, OUTPUT_ROOT, K_19_PATH, normalize_input, load_tif_windows, BERN_TILES_ROOT
from facebookresearch_dino_main.eval_knn import ReturnIndexDataset
import facebookresearch_dino_main.utils as utils
from glob import glob


@torch.no_grad()
def save_features(model:torch.nn.Module, data_loader:DataLoader, out_dir, multiscale=False, save_coords=True):
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
    # time.sleep(0.01)
    if save_coords:
        coords = torch.zeros(len(data_loader.dataset), 2).cpu().long()
        for samples, index, fnames in tqdm(data_loader, desc=f'Feature extraction for {os.path.basename(out_dir)}...'):
            coords_x = []
            coords_y=[]
            for filename in fnames:
                # print(filename)
                filename = os.path.basename(filename)
                split = filename.split(".")[2].split("_")
                coords_x.append(int(split[2][1:]))
                coords_y.append(int(split[3][1:]))
            samples = samples.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)
            coords[index] = torch.vstack((torch.tensor(coords_x), torch.tensor(coords_y))).T.long()
            labels = torch.tensor([s[-1] for s in data_loader.dataset.samples]).long()
            if multiscale:
                feats = utils.multi_scale(samples, model)
            else:
                feats = model(samples).clone()
            features[index] = feats.cpu()
    else:
        for samples, index in tqdm(data_loader, desc=f'Feature extraction for {os.path.basename(out_dir)}...'):
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
    if save_coords:
        torch.save(coords, os.path.join(out_dir, 'coords.pt'))

if __name__ == '__main__':
    # data = get_data_loader(224, 16, 64, dataset_class=ReturnIndexDataset, shuffle=False)
    t = normalize_input(im_size=224, patch_size=8)
    # transform = tr.Compose([t,
    #                         tr.RandomHorizontalFlip(p=1)])
    transform = t
    model = get_vit(8, pretrained_weight_path='ckpts/dino_vits8_tiny_full_k19.pth', arch='vit_tiny', key='teacher')
    # for roi in glob(os.path.join(BERN_TILES_ROOT, '*')):
        # print(roi)
        # break

    ds = Abed_utils.ReturnIndexDataset_K19(Abed_utils.K_19_VAL_PATH, transform=transform, loader=load_tif_windows, return_fname=False, is_valid_file=lambda x: x.endswith('tif'))
    # print(ds.classes)
    # exit(0)
    #     ds = Abed_utils.ReturnIndexDataset_K19(roi, transform=transform, loader=load_tif_windows,
    #                                            return_fname=True)
    data = DataLoader(ds, 64, num_workers=4)

    save_features(model, data, out_dir=os.path.join(OUTPUT_ROOT, f'features-trained-val'), save_coords=False)
