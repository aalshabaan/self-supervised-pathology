import Abed_utils

import os
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Tuple, Union
from glob import glob
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt

class ROIDataset(Dataset):
    def __init__(self, path_to_image:str, patch_size:int=224, padding_factor:float=0.5, loader=None, transform=None):
        self.patch_size = patch_size
        self.step_size = padding_factor
        self.transform = transform

        self.im = Image.open(path_to_image) if loader is None else loader(path_to_image)

        self.center_grid = ROIDataset._build_reference_grid(patch_size, padding_factor, self.im.size)

    def __getitem__(self, item:Union[int, slice]) -> Tuple[Image.Image, List[int]]:
        """
        Returns a crop from the opened image along with its size and coordinates.
        :param item : int|slice: index of the chosen crop
        :return: tuple of the chosen crop and a list of its size and coordinates
        """
        if not isinstance(item, (int, slice)):
            raise TypeError('Index must be int or slice')
        x, y = self.center_grid[item, :]
        m = self.patch_size//2
        crop = self.im.crop((x-m, y-m, x+m, y+m))
        crop = self.transform(crop)

        return crop, [self.patch_size, x, y, x-m, y-m, x+m, y+m]

    def __len__(self):
        return self.center_grid.shape[0]

    @staticmethod
    def _build_reference_grid(
            crop_size_px: int,
            padding_factor: float,
            im_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Build reference grid for cropping location.
        Parameters
        ----------
        crop_size_px: int
            Output size in pixel.
        padding_factor: float
            Padding factor to use. Define the interval between two consecutive crops.
        im_size: list of int
            Size of the image to crop from.
        Returns
        -------
        (cx, cy): list of int
            Center coordinate of the crop.
        """

        # Define the size of the crop at the selected level
        # crop_size_px = int((level_magnification / crop_magnification) * crop_size_px)

        # Compute the number of crops for each dimensions (rows and columns)
        n_w = np.floor((1 / padding_factor) * (im_size[0] / crop_size_px - 1)).astype(int)
        n_h = np.floor((1 / padding_factor) * (im_size[1] / crop_size_px - 1)).astype(int)

        # Compute the residual margin at each side of the image
        margin_w = int(im_size[0] - padding_factor * (n_w - 1) * crop_size_px) // 2
        margin_h = int(im_size[1] - padding_factor * (n_h - 1) * crop_size_px) // 2

        # Compute the final center for the cropping
        c_x = (np.arange(n_w) * crop_size_px * padding_factor + margin_w).astype(int)
        c_y = (np.arange(n_h) * crop_size_px * padding_factor + margin_h).astype(int)
        c_x, c_y = np.meshgrid(c_x, c_y)

        return np.array([c_x.flatten(), c_y.flatten()]).T

def parse_roi_name(name:str) -> str:
    return os.path.basename(name).split('.png')[0]

def parse_args():
    parser = ArgumentParser(description='Segment an ROI by classifying patches.')
    parser.add_argument('--patch-size', default=224, help='Size of the sliding window.', type=int)
    parser.add_argument('--padding', default=0.5, help='Padding factor to build the prediction grid.', type=float)
    parser.add_argument('--roi-path', help='The path towards the ROI images. should contain .png images', required=True)
    parser.add_argument('--cuda-dev', help='The cude device to use if specified', default=None)
    parser.add_argument('--batch-size', help='The batch size to be used', default=64, type=int)
    parser.add_argument('--out-subdir', help='Subdirectory of the output folder', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    files = glob(os.path.join(args.roi_path, '*.png'))
    device = torch.device(f'cuda:{args.cuda_dev}') if torch.cuda.is_available() and args.cuda_dev is not None\
        else torch.device('cpu')
    backbone = Abed_utils.get_vit(8, './ckpts/dino_deitsmall8_pretrain.pth', arch='vit_small', device=device)
    features, labels = Abed_utils.load_features(os.path.join(Abed_utils.OUTPUT_ROOT, 'features'), device=device)
    classifier = Abed_utils.KNNClassifier(features, labels)

    model = nn.Sequential(backbone, classifier)
    model.eval()

    TUM = 8
    STR = 7
    outpath = os.path.join(Abed_utils.OUTPUT_ROOT, args.out_subdir)
    os.makedirs(outpath, exist_ok=True)

    tsrs = {}
    with torch.inference_mode():
        for f in files:
            all_preds = []
            all_metas = []
            wsi_name = parse_roi_name(f)
            ds = ROIDataset(f, patch_size=args.patch_size, padding_factor=args.padding, transform=Abed_utils.normalize_input(args.patch_size, 8))
            loader = DataLoader(ds, shuffle=False, batch_size=args.batch_size, drop_last=False, num_workers=4)
            for x, metas in tqdm(loader, desc=f'Classifying roi of {wsi_name}...'):
                all_metas.extend(torch.vstack(metas).T)
                preds = model(x.to(device))
                all_preds.extend(preds.cpu())

            all_preds = np.array(all_preds)
            T = (all_preds == TUM).sum()
            S = (all_preds == STR).sum()
            tsrs[wsi_name] = T/(T+S)
            df = pd.DataFrame(data=all_metas, columns=['s_src', 'cx', 'cy', 'tx', 'ty', 'bx', 'by'])
            df = df.applymap(lambda x: x.item() if isinstance(x, torch.Tensor) else x)
            df['classification'] = all_preds

            df.to_csv(os.path.join(outpath, f'{wsi_name}.csv'), index=False)

    pd.Series(tsrs).astype(float).to_csv(os.path.join(outpath, 'TSR.csv'))
