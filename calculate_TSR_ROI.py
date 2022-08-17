import os.path

import Abed_utils
import pandas as pd
import numpy as np
from os.path import join
from glob import glob
from tqdm import tqdm
# import warnings
# warnings.filterwarnings('error')

predictions_path = join(Abed_utils.OUTPUT_ROOT, 'predictions_KNN')
roi_path = join(Abed_utils.OUTPUT_ROOT, 'ROI_detection_final_full')

def basename_from_roi(filename:str):
    name = os.path.basename(filename)
    return name.split('_roi')[0]

tsrs = {}
def main():
    for roi_filename in tqdm(glob(join(roi_path, '*.csv'))):
        basename = basename_from_roi(roi_filename)
        pred_filename = f'{basename}_seg_dino_imagenet_100ep_KNN.npy'
        pred_dict = np.load(join(predictions_path, pred_filename), allow_pickle=True).item()

        df = pd.DataFrame(pred_dict['metadata'], columns=pred_dict['metadata_labels'])
        df['classification'] = pred_dict['classification']
        possible_rois = pd.read_csv(roi_filename, index_col='Unnamed: 0').squeeze('columns')
        # print(possible_rois)
        roi_coords = possible_rois[['x', 'y']]
        # print(f'ROI: {roi_coords}')
        radius = 1250/possible_rois.mpp
        mask = np.sqrt((df.cx - roi_coords.x)**2 + (df.cy-roi_coords.y)**2) <= radius
        # print(f'MASK SUM: {mask.sum()}')

        roi_preds = df.loc[mask, 'classification']
        print(roi_preds.shape)
        # break

        T = (roi_preds == 8).sum()
        S = (roi_preds == 7).sum()

        if T or S:
            tsrs[basename] = T / (T + S)
        else:
            print(f'for {basename}: T:{T}, S:{S}, considered samples: {roi_preds.shape[0]}')
            tsrs[basename] = np.nan

    # pd.Series(tsrs, dtype=float).to_csv(join(Abed_utils.OUTPUT_ROOT, 'ROI_detection_final_full.csv'))

if __name__ == '__main__':
    main()