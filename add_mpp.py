import os

import Abed_utils

import pandas as pd
from glob import glob
from tqdm import tqdm

from wsi import WholeSlideDataset

if __name__ == '__main__':
    for csv in tqdm(glob(os.path.join(Abed_utils.OUTPUT_ROOT, 'ROI_detections', '*.csv'))):
        bname = os.path.basename(csv).split('_roi')[0]
        print(csv, bname)
        dataset = WholeSlideDataset(os.path.join(Abed_utils.BERN_COHORT_ROOT, bname))
        df = pd.read_csv(csv)
        df['mpp'] = dataset.mpp

        df.to_csv(csv, index=False)