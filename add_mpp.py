import os

import Abed_utils

import pandas as pd
from glob import glob
from tqdm import tqdm

from wsi import WholeSlideDataset

for csv in tqdm(glob(os.path.join(Abed_utils.OUTPUT_ROOT, 'ROI_detections'))):
    bname = os.path.basename(csv).split('_roi')[0]
    dataset = WholeSlideDataset(os.path.join(Abed_utils.BERN_COHORT_ROOT, bname))
    df = pd.read_csv(csv)
    df['mpp'] = dataset.mpp

    df.to_csv(csv, index=False)