import os

import Abed_utils

import pandas as pd
from glob import glob
from tqdm import tqdm
import re
from argparse import ArgumentParser
from wsi import WholeSlideDataset

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('output_subdir', default='ROI_detections')
    args = parser.parse_args()
    for csv in tqdm(glob(os.path.join(Abed_utils.OUTPUT_ROOT, args.output_subdir, '*.csv'))):
        bname = os.path.basename(csv).split('_roi')[0]
        dir_name = str(int(re.match(r'^\d+', bname)[0]))
        # print(dir_name, bname)
        dataset = WholeSlideDataset(os.path.join(Abed_utils.BERN_COHORT_ROOT, dir_name, bname))
        df = pd.read_csv(csv)
        df['mpp'] = dataset.mpp

        df.to_csv(csv, index=False)