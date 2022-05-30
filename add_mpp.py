import os

import Abed_utils

import pandas as pd
from glob import glob
from tqdm import tqdm
import re

from wsi import WholeSlideDataset

if __name__ == '__main__':
    for csv in tqdm(glob(os.path.join(Abed_utils.OUTPUT_ROOT, 'ROI_detections', '*.csv'))):
        bname = os.path.basename(csv).split('_roi')[0]
        dir_name = re.match(r'^\d+', bname)[0]
        print(dir_name)
        dataset = WholeSlideDataset(os.path.join(Abed_utils.BERN_COHORT_ROOT, dir_name, bname))
        df = pd.read_csv(csv)
        df['mpp'] = dataset.mpp

        df.to_csv(csv, index=False)