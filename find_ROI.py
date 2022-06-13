
import Abed_utils

from wsi import WholeSlideDataset
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import pandas as pd
from PIL import Image
from PIL.ImageDraw import ImageDraw
from glob import glob
from tqdm import tqdm
from scipy.signal import correlate

from argparse import ArgumentParser



# @torch.no_grad()
# def get_valid_points(filter:Image.Image, preds:torch.Tensor) -> torch.Tensor:


@torch.no_grad()
def main(args):
    output = []
    outpath = os.path.join(Abed_utils.OUTPUT_ROOT, args.out_subdir if args.out_subdir is not None else f'ROI_detections_p{args.p}')
    roi_path = os.path.join(outpath, 'roi')
    os.makedirs(roi_path, exist_ok=True)

    device = torch.device(f'cuda:{args.cuda_device}') if torch.cuda.is_available() and args.cuda_device is not None\
        else torch.device('cpu')

    for path in tqdm(glob(os.path.join(Abed_utils.BERN_COHORT_ROOT, '*', '*.mrxs'))):

        bname = os.path.basename(path)+'.png'

        if os.path.isfile(os.path.join(roi_path, bname)):
            print(f'{bname} already calculated, skipping!')
            continue

        # Load WSI (for some metadata) and predictions
        wsi = WholeSlideDataset(path)

        preds = np.load(os.path.join(Abed_utils.OUTPUT_ROOT,
                                     'predictions_KNN',
                                     f'{os.path.basename(wsi.path)}_seg_dino_imagenet_100ep_KNN.npy'),
                        allow_pickle=True).item()
        # Fix constants and factors
        pred_patch_size = int(preds['metadata'][0, -2])
        if (preds['metadata'][:, -2] == pred_patch_size).all():
            # print(f'Each patch covers {pred_patch_size}px^2 of the WSI')
            pass
        else:
            raise RuntimeError('Not all patches are of the same size!')

        downsample_factor = 1

        patch_size = wsi.mpp * downsample_factor * pred_patch_size  # [um/patch]
        diameter = round(2500 / patch_size)  # [patch]
        search_radius = round(500 / patch_size)  # [patch]

        # Build convolution kernel
        p = args.p  # Percentage of the circle to show at each side

        kernel = Image.new('1', (diameter, diameter), 0)
        draw = ImageDraw(kernel)
        draw.ellipse(xy=[((p - 1) * diameter, 0), (p * diameter, diameter)], fill=1)
        kernels = [kernel]
        rotations = [90, 180, 270]
        for rot in rotations:
            kernels.append(kernel.rotate(rot))

        kernel = Image.new('1', kernel.size, 1)
        # draw = ImageDraw(kernel)
        # draw.ellipse(((0,0), (diameter, diameter)), fill=1)
        kernels.append(kernel)

        # Build prediction tensor
        pred_tensor = torch.zeros(wsi.s.dimensions[1] // (pred_patch_size * downsample_factor),
                                  wsi.s.dimensions[0] // (pred_patch_size * downsample_factor))
        # print(f'Prediction tensor is of size {pred_tensor.shape}')

        metadata = preds['metadata'].astype(int)
        df = pd.DataFrame(data=metadata[:, 2:8], columns=preds['metadata_labels'][2:8])
        df = df // (pred_patch_size * downsample_factor)
        df['pred'] = preds['classification'].astype(int)

        # Put 1 wherever we have a tumor, -1 for Stroma, and 0 for everything else
        for coords, group in df.groupby(by=['cx', 'cy']):
            # print(group.pred)
            # if len(group.pred.mode()) > 1:
            #     break
            modes = group.pred.mode()
            if 7 in modes.values:
                pred_tensor[coords[::-1]] = -1
            elif 8 in modes.values:
                pred_tensor[coords[::-1]] = 1
            else:
                pred_tensor[coords[::-1]] = 0

        pred_tensor = pred_tensor.to(device)
        rotations = range(0, 50, 5)
        # fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        # fig.suptitle('Rotations')
        vmax = -np.inf
        xmax, ymax = None, None

        for i, rot in enumerate(rotations):
            k = torch.concat([T.ToTensor()(x.rotate(rot)) for x in kernels]).unsqueeze(1).to(device)
            # print(pred_tensor.shape)
            # Put the stroma filter to -1
            # k[-1, :, :, :] *= -1
            # Convolve for heatmap
            # hmaps = F.conv2d(input=pred_tensor.unsqueeze(0), weight=k, stride=(1,), padding='same',
            #                  dilation=(1,)).relu()
            # mask = (hmaps[:4, :, :] > 0).sum(0) == 4
            # # hmap = hmaps.mean(0)
            # hmap = hmaps[4,:,:]
            # hmap[~mask] = 0
            mask = (F.conv2d(input=(pred_tensor.unsqueeze(0) == 1).float(), weight=k[:4, :, :], stride=(1,),
                             padding='same', dilation=(1,)) > 0).sum(0) == 4
            hmap = F.conv2d(input=(pred_tensor.unsqueeze(0) == -1).float(), weight=k[4, :, :].unsqueeze(0), stride=(1,),
                            padding='same', dilation=(1,)).squeeze()
            hmap[~mask] = 0

            v, idxs = hmap.flatten().topk(1)
            x, y = idxs.item() % hmap.shape[1], torch.div(idxs, hmap.shape[1], rounding_mode='floor').item()

            if v > vmax:
                vmax = v.item()
                xmax, ymax = x, y

        output.append([xmax, ymax, vmax, wsi.mpp, pred_patch_size])
        # ds_idx = len(wsi.level_downsamples) -1
        ds_idx = 0
        ds = int(wsi.level_downsamples[ds_idx])
        wsi.s.read_region(((xmax-diameter//2)*pred_patch_size,(ymax-diameter//2)*pred_patch_size), ds_idx, ((diameter * pred_patch_size)//ds, (diameter * pred_patch_size)//ds)).convert('RGB')\
            .save(os.path.join(roi_path, bname))

        # coords = pd.DataFrame(data=output, columns=['x', 'y', 'value', 'mpp', 'patch_size'])
        coords = pd.Series(data=[xmax, ymax, vmax, wsi.mpp, pred_patch_size], index=['x', 'y', 'value', 'mpp', 'patch_size'])
        # coords = coords.applymap(lambda x: x.item())
        coords[['x', 'y']] *= pred_patch_size * downsample_factor
        # coords['mpp'] = wsi.mpp
        coords.to_csv(os.path.join(outpath, f'{bname}.csv'), encoding='UTF-8', index=False)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--out_subdir', default=None)
    parser.add_argument('-p', default=0.15, type=float)
    parser.add_argument('--cuda-device', default=None, type=int)
    args = parser.parse_args()
    main(args)