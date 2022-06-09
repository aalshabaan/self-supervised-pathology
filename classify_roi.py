import Abed_utils

import os
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import IterableDataset
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt

class ROIDataset(IterableDataset):
    def __init__(self, path_to_image:str, patch_size:int=224, step_size:float=0.5):
        self.patch_size = patch_size
        self.step_size = step_size



def segment_roi(roi:Image.Image, model:torch.nn.Module, patch_size:int=224, overlap:float=0.5):
    step = patch_size*overlap



if __name__ == '__main__':
    parser = ArgumentParser(description='Segment an ROI by classifying patches')
