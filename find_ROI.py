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


def main():
    outpath = os.path.join(Abed_utils.OUTPUT_ROOT, 'ROI_detections_expanded')
    os.makedirs(outpath, exist_ok=True)

    for path in tqdm(glob(os.path.join(Abed_utils.BERN_COHORT_ROOT, '*', '*.mrxs'))):
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
        p = 0.15  # Percentage of the circle to show at each side

        kernel = Image.new('L', (diameter, diameter), 0)
        draw = ImageDraw(kernel)
        draw.ellipse(xy=[((p - 1) * diameter, 0), (p * diameter, diameter)], fill=255)
        draw.ellipse(xy=[((1 - p) * diameter, 0), ((2 - p) * diameter, diameter)], fill=255)
        draw.ellipse(xy=[(0, (p - 1) * diameter), (diameter, p * diameter)], fill=255)
        draw.ellipse(xy=[(0, (1 - p) * diameter), (diameter, (2 - p) * diameter)], fill=255)

        kernel = kernel.convert('F')

        # Build prediction tensor
        pred_tensor = torch.zeros(wsi.s.dimensions[1] // (pred_patch_size * downsample_factor),
                                  wsi.s.dimensions[0] // (pred_patch_size * downsample_factor), dtype=torch.uint8)
        # print(f'Prediction tensor is of size {pred_tensor.shape}')

        metadata = preds['metadata'].astype(int)
        df = pd.DataFrame(data=metadata[:, 2:8], columns=preds['metadata_labels'][2:8])
        df = df // (pred_patch_size * downsample_factor)
        df['pred'] = preds['classification'].astype(int)

        # Put 1 wherever we have a tumor, -1 for Stroma, and 0 for everything else
        for coords, group in df.groupby(by=['cx', 'cy']):
            pred_tensor[coords[::-1]] = group.pred.mode()[0]

        pred_tensor = pred_tensor.float()
        for i in range(7):
            pred_tensor[pred_tensor == i] = 0
        pred_tensor[pred_tensor == 7] = -1
        pred_tensor[pred_tensor == 8] = 1

        rotations = range(0, 50, 5)
        # fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        # fig.suptitle('Rotations')
        vals = []
        xs, ys = [], []
        with torch.no_grad():
            for i, rot in enumerate(rotations):
                k = kernel.rotate(rot, expand=True)
                k = T.ToTensor()(k).unsqueeze(0) / 255
                # print(pred_tensor.shape)
                k = (2 * k - 1)
                # print(f'Kernel: {torch.unique(k)}, {k.shape}, {k.dtype}')
                # print(k.shape)
                # print(f'Input: {torch.unique(pred_tensor)}, {pred_tensor.shape}, {pred_tensor.dtype}')
                hmap = F.conv2d(input=pred_tensor.unsqueeze(0).float(), weight=k, stride=(1,), padding='valid',
                                dilation=(1,))[0, :, :]

                v, idxs = hmap.flatten().topk(1)
                x, y = idxs % hmap.shape[1], idxs // hmap.shape[1]
                xs.extend(x)
                ys.extend(y)
                vals.extend(v)

                # ax = axs[i//3, i%3]
                # ax.imshow(hmap.relu())
                # ax.set_title(f'{rot} degrees')
                # ax.grid(None)
                # ax.set_xticks([])
                # ax.set_yticks([])
        coords = pd.DataFrame(data=list(zip(xs, ys, vals)), columns=['x', 'y', 'value'])
        coords = coords.applymap(lambda x: x.item())
        coords[['x', 'y']] *= pred_patch_size * downsample_factor

        coords.to_csv(os.path.join(outpath, f'{os.path.basename(path)}_roi.csv'), encoding='UTF-8')





if __name__ == '__main__':
    main()