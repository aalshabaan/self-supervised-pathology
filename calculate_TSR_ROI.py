import os.path

import Abed_utils
import pandas as pd
import numpy as np
from os.path import join
from glob import glob
from tqdm import tqdm

predictions_path = join(Abed_utils.OUTPUT_ROOT, 'predictions_KNN')
roi_path = join(Abed_utils.OUTPUT_ROOT, 'ROI_detections')

def basename_from_roi(filename:str):
    name = os.path.basename(filename)
    return name.split('_roi')[0]


def main():
    for roi_filename in tqdm(glob(join(roi_path, '*.csv'))):
        basename = basename_from_roi(roi_filename)
        pred_filename = f'{basename}__seg_dino_imagenet_100ep_KNN.npy'
        pred_dict = np.load(join(predictions_path, pred_filename), allow_pickle=True).item()

        df = pd.DataFrame(pred_dict['metadata'], columns=pred_dict['metadata_labels'])
        df['classification'] = pred_dict['classification']
        possible_rois = pd.read_csv(roi_filename)
        roi_coords = possible_rois.sort_values(by='value', ascending=False)[['x', 'y']]
        radius = 2500
        if not (df.s_src == patch_size).all():
            raise RuntimeError('Not all patch sizes are the same!')
        mask = np.logical_and(np.abs(df.cx - roi_coords.x) < patch_size/2, np.abs(df.cy - roi_coords.y) < patch_size/2)

        roi_preds = df.loc[mask, ['classification']]

if __name__ == '__main__':
    main()