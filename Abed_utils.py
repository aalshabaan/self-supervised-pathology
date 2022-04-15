import functools
import logging
import math
import os
import sys
from platform import node
from typing import Union, Tuple
import json

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from skimage.io import imread
from PIL import Image, ImageDraw, ImageFont
import numpy as np

sys.path.extend([os.getcwd(), os.path.join(os.getcwd(), 'facebookresearch_dino_main')])

from facebookresearch_dino_main.vision_transformer import vit_small
from wsi import WholeSlideDataset


def my_pc():
    return 'Abdulrahman' in node()

K_19_PATH = 'D:/self_supervised_pathology/datasets/NCT-CRC-HE-100K-NONORM'\
    if my_pc() else '/mnt/data/dataset/tiled/kather19tiles_nonorm/NCT-CRC-HE-100K-NONORM'

OUTPUT_ROOT = 'D:/self_supervised_pathology/output/'\
    if my_pc() else '/home/guest/Documents/shabaan_2022/output/'

TEST_SLIDE_PATH = "D:/self_supervised_pathology/datasets/WSI/TCGA-CK-6747-01Z-00-DX1.7824596c-84db-4bee-b149-cd8f617c285f.svs"\
    if my_pc() else None

BERN_COHORT_ROOT = r"D:\self_supervised_pathology\datasets\WSI\bern_cohort_clean"\
    if my_pc() else '/mnt/data/dataset/bern_cohort_clean/'

DATETIME_FORMAT = '%m-%d_%H:%M:%S'

output_paths = []


_model = None
class NamedImageFolder(ImageFolder):
    def __getitem__(self, item):
        return self.transform(self.loader(self.imgs[item][0])), *self.imgs[item]

def get_model(patch_size, pretrained_weight_path, key=None, device='cuda'):
    if patch_size not in [8, 16]:
        raise ValueError('patch size must be 8 or 16')
    global _model
    if _model is None:
        _model = vit_small(patch_size=patch_size, num_classes=0)
        state_dict = torch.load(pretrained_weight_path)
        if key is not None:
            state_dict = state_dict[key]
            to_remove = []
            for key in state_dict.keys():
                if 'head' in key:
                    to_remove.append(key)
            for key in to_remove:
                state_dict.pop(key)
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = _model.load_state_dict(state_dict)
        print(f'state dict loaded with the message {msg}')
        _model = _model.to(device)
    return _model


def normalize_input(input, im_size, patch_size):
    # print(type(input), input)
    t = transforms.Compose([transforms.ToTensor(), transforms.Resize(im_size), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    img = t(input)

    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h]

    return img

def get_data_loader(im_size, patch_size, batch_size=1, whole_slide=False, output_subdir=None, dataset_class=NamedImageFolder, shuffle=True):
    global output_paths
    global OUTPUT_ROOT
    output_subdir = os.path.join(OUTPUT_ROOT, output_subdir) if output_subdir is not None else OUTPUT_ROOT
    t = functools.partial(normalize_input, im_size=im_size, patch_size=patch_size)
    if whole_slide:
        if TEST_SLIDE_PATH is None:
            raise ValueError('Whole slide image is not available on this machine')
        ds = WholeSlideDataset(TEST_SLIDE_PATH, transform=t, )
        slide_name = os.path.split(TEST_SLIDE_PATH)[1].split(os.extsep)[0]
        output_paths.append(os.path.join(output_subdir, slide_name))
    else:
        ds = dataset_class(K_19_PATH, t, loader=load_tif_windows)
        for i, cls in enumerate(ds.classes):
            output_paths.append(os.path.join(output_subdir, cls))
            # os.makedirs(output_paths[i], exist_ok=True)
    dl = DataLoader(ds, batch_size, shuffle=shuffle)
    return dl

def load_tif_windows(im_path:str):
    return Image.fromarray(imread(im_path))

@torch.no_grad()
def distance_matrix(x, y=None, p = 2): #pairwise distance of vectors

    y = x if type(y) == type(None) else y
    x = x.cpu()
    y = y.cpu()

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2)

    return dist

@torch.no_grad()
def knn_classifier(train_features:torch.Tensor, train_labels, test_features, k:int):

    num_test_images, num_chunks = test_features.shape[0], 100
    imgs_per_chunk = max(num_test_images // num_chunks, num_test_images)
    preds = torch.zeros(test_features.shape[0], device=torch.device('cpu'))

    for idx in range(0, num_test_images, imgs_per_chunk):
        features = test_features[idx:min(idx+imgs_per_chunk, num_test_images), :].cpu()
        dists = distance_matrix(features, train_features)
        _, voters = torch.topk(dists, k, largest=False)
        votes = torch.gather(train_labels.view(1, -1).expand(features.shape[0], -1),1, voters)
        batch_preds, _ = torch.mode(votes, 1)
        preds[idx : min((idx + imgs_per_chunk), num_test_images)] = batch_preds

    return preds

@torch.no_grad()
def similarity_knn_classifier(train_features:torch.Tensor, train_labels:torch.Tensor, test_features:torch.Tensor, k:int):

    num_test_images, num_chunks = test_features.shape[0], 100
    imgs_per_chunk = max(num_test_images // num_chunks, num_test_images)
    train_feats_transpose = train_features.t()

    preds = torch.zeros(test_features.shape[0], device=test_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[idx:min((idx+imgs_per_chunk), num_test_images), :]
        similarities = features.mm(train_feats_transpose).t().div(features.norm(2,1)).t().div(train_features.norm(2,1))
        _, voters = torch.topk(similarities, k, 1)
        votes = torch.gather(train_labels.view(1, -1).expand(features.shape[0], -1),1, voters)
        batch_preds, _ = torch.mode(votes, 1)
        preds[idx : min((idx + imgs_per_chunk), num_test_images)] = batch_preds

    return preds


@torch.no_grad()
def weighted_similarity_knn_classifier(train_features, train_labels, test_features, k, T, num_classes=9):

    # top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_features.shape[0], 100
    imgs_per_chunk = max(num_test_images // num_chunks, num_test_images)
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    preds = torch.zeros(test_features.shape[0], device=test_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        batch_size = features.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)
        # return predictions
        # print(predictions.shape)
        preds[idx:min(idx+imgs_per_chunk, num_test_images)] = predictions[:,-1]

        # find the predictions that match the target
        # correct = predictions.eq(targets.data.view(-1, 1))
        # top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        # top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        # total += targets.size(0)
    return preds
    # top1 = top1 * 100.0 / total
    # top5 = top5 * 100.0 / total
    # return top1, top5

class KNNClassifier(nn.Module):
    def __init__(self, data, lbl, k=20, mode='cos'):
        self.data = data
        self.labels = lbl
        if k in range(1, self.data.shape[0]+1):
            self.k = k
        else:
            raise ValueError('K must be between 1 and the total number of training samples')
        if mode.casefold() in ['cos', 'dist', 'weighted']:
            self.mode = mode.lower()
        else:
            raise ValueError('Invalid KNN mode')

    def __call__(self, x):
        if self.mode == 'cos':
            return similarity_knn_classifier(self.data, self.labels, x, self.k)
        elif self.mode == 'weighted':
            return knn_classifier(self.data, self.labels, x, self.k)
        else:
            return weighted_similarity_knn_classifier(self.data, self.labels, x, self.k)


def load_features(path, cuda=False, load_labels=True):
    device = torch.device('cuda') if cuda and torch.cuda.is_available() else torch.device('cpu')
    features = torch.load(os.path.join(path, 'features.pt')).to(device)
    labels = torch.load(os.path.join(path, 'labels.pt')).to(device) if load_labels else None

    return features, labels


def load_wsi(path_to_wsi, crop_size, patch_size):
    t = functools.partial(normalize_input, im_size=crop_size, patch_size=patch_size)
    ds = WholeSlideDataset(path_to_wsi, transform=t, crop_sizes_px=[crop_size])

    return ds


def build_poly(tx: np.ndarray, ty: np.ndarray, bx: np.ndarray, by: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Counter clock-wise
    px = np.vstack((tx, bx, bx, tx)).T
    py = np.vstack((ty, ty, by, by)).T

    return px, py


def save_annotation_qupath(
        tx: np.ndarray,
        ty: np.ndarray,
        bx: np.ndarray,
        by: np.ndarray,
        values: np.ndarray,
        values_name: Union[np.ndarray, dict],
        outpath: str,
        cmap: Colormap,
) -> None:
    """
    Parameters
    ----------
    tx: array_like
    ty: array_like
    bx: array_like
    by: array_like
    values: array_like
    values_name: array_like
    outpath: str
    cmap: Colormap
    """

    # Check dimensions
    if not all(tx.shape == np.array([ty.shape, bx.shape, by.shape, values.shape])):
        return

    # Build shape and simplify the shapes if True
    polys_x, polys_y = build_poly(tx=tx, ty=ty, bx=bx, by=by)

    # Extract outer shapes
    coords = {}
    colors = []
    clss = []
    for i in range(len(polys_x)):
        color = 255*np.array(cmap(values[i]))[:3]
        colors.append(color)
        label = values[i] if isinstance(values_name, np.ndarray) else values_name[values[i]]
        clss.append([label])
        coords['poly{}'.format(i)] = {
            "coords": np.vstack((polys_x[i], polys_y[i])).tolist(),
            "class": str(label),
            "color": [int(color[0]), int(color[1]), int(color[2])]
        }

    with open(outpath, 'w') as outfile:
        json.dump(coords, outfile)


class ClassificationHead(nn.Module):
    # A simple MLP, only reason this class exists is to ensure a compatible MLP architecture across different scripts.
    logger = logging.getLogger('load_mlp')

    def __init__(self, in_dim=384, hidden_dim=100, out_dim=9, pretrained_path=None, device='cuda'):
        super().__init__()
        # self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=False),
        #                          nn.ReLU(),
        #                          nn.Linear(hidden_dim, out_dim, bias=False)).to(device)

        self.mlp = nn.Sequential(nn.Dropout(0.2),nn.Linear(in_dim, out_dim)).to(device)
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path)
            msg = self.load_state_dict(state_dict)
            self.logger.info(f'Loaded MLP state dict with message {msg}')
        else:
            self.logger.info(f'No pretrained MLP given, loading random weights')

    def forward(self, x):
        return self.mlp(x)
