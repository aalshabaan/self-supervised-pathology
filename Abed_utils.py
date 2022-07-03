import functools
import logging
import math
import os
import sys
from platform import node
from typing import Union, Tuple, List, Dict, Optional, Callable
import json

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, ListedColormap
from matplotlib import cm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from skimage.io import imread
from PIL import Image, ImageDraw, ImageFont
import numpy as np

sys.path.extend([os.getcwd(), os.path.join(os.getcwd(), 'facebookresearch_dino_main')])

from facebookresearch_dino_main.vision_transformer import vit_small, vit_tiny
from wsi import WholeSlideDataset


def my_pc():
    return 'Abdulrahman' in node()

K_19_PATH = 'D:/self_supervised_pathology/datasets/NCT-CRC-HE-100K-NONORM'\
    if my_pc() else '/mnt/data/dataset/tiled/kather19tiles_nonorm/NCT-CRC-HE-100K'

OUTPUT_ROOT = 'D:/self_supervised_pathology/output/'\
    if my_pc() else '/home/guest/Documents/shabaan_2022/output/'

TEST_SLIDE_PATH = "D:/self_supervised_pathology/datasets/WSI/TCGA-CK-6747-01Z-00-DX1.7824596c-84db-4bee-b149-cd8f617c285f.svs"\
    if my_pc() else None

BERN_COHORT_ROOT = r"D:\self_supervised_pathology\datasets\WSI\bern_cohort_clean"\
    if my_pc() else '/mnt/data/dataset/bern_cohort_clean/'

BERN_TILES_ROOT = r'D:\self_supervised_pathology\datasets\bern_crop_label' if my_pc()\
    else '/mnt/data/dataset/tiled/bern_crop_label'

DATETIME_FORMAT = '%m-%d_%H:%M:%S'

output_paths = []

_model = None


class NamedImageFolder(ImageFolder):
    def __getitem__(self, item):
        return self.transform(self.loader(self.imgs[item][0])), *self.imgs[item]


class BernTilesLabelDataset(ImageFolder):
    """
    A custom Dataset class that reads all images from a folder and infers the class label from the filenames.
    Assumed filename structure is lbl_filename.ext where "lbl" is the class label.
    """

    def __init__(self, *args, return_index=False, **kwargs):
        self.return_index = return_index
        super().__init__(*args, **kwargs)

    #TODO: add a RegEx argument to match the label for any pattern?
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        # classes = sorted(torch.unique([x.split('_')[0] for x in os.listdir(directory) if os.path.isfile(os.path.join(directory, x))]))
        classes = sorted(['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'])
        return classes, {c: i for i, c in enumerate(classes)}

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        if not class_to_idx:
            raise ValueError('Unknown mapping from class to target')
        if extensions is not None and is_valid_file is not None:
            raise ValueError('Can\'t pass both extensions and is_valid_file')

        if is_valid_file is not None:
            crit = is_valid_file
        elif extensions is not None:
            crit = lambda x: os.path.splitext(x)[1] in extensions
        else:
            crit = lambda x: True

        files = [x for x in os.listdir(directory) if crit(x)]
        targets = [class_to_idx[x.split('_')[0]] for x in files]

        return list(zip(files, targets))


class ReturnIndexDataset_K19(ImageFolder):
    """
    Image folder subclass that returns (datapoint, index) instead of (datapoint, label). Labels are assumed to be Kather-19 labels
    """
    def __init__(self, *args, return_fname=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_fname=return_fname

    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset_K19, self).__getitem__(idx)
        if self.return_fname:
            fname, _ = self.samples[idx]
            return img, idx, fname
        return img, idx

    def find_classes(self, directory: str):
        classes, _ = super(ReturnIndexDataset_K19, self).find_classes(directory)
        classes_to_map = sorted(['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'])
        return classes, {c: classes_to_map.index(c) for c in classes}


def get_vit(patch_size, pretrained_weight_path, key=None, device='cuda', arch='vit_small'):
    """
    Build a vision transformer (as defined in the DINO code repo), load a checkpoint and return it
    :param patch_size: Patch size for the vision transformer. An image will be split into multiple patches of this size and passed into the transformer as a bag of patches
    :param pretrained_weight_path: Path towards the file containing the pretrained weights
    :param key: Key from the dictionary that contains the state dict with pretrained weights, if not given the base dict is assumed to be the state dict
    :param device: Device on which the model will be loaded. 'cuda' by default
    :return: The loaded model
    """
    if patch_size not in [8, 16]:
        raise ValueError('patch size must be 8 or 16')
    global _model
    if _model is None:
        if arch == 'vit_small':
            _model = vit_small(patch_size=patch_size, num_classes=0)
        elif arch == 'vit_tiny':
            _model = vit_tiny(patch_size=patch_size, num_classes=0)
        else:
            raise ValueError(f'Unsupported architecture {arch}!')
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

def _normalize(input, im_size, patch_size):
    t = transforms.Compose([transforms.ToTensor(), transforms.Resize(im_size), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    img = t(input)

    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h]

    return img

def normalize_input(im_size, patch_size):
    """
    Returns a function that takes a PIL image (or a batch thereof) and returns them as tensors resized to the specified size
    and being divisible into patches of patch_size.
    This last part is done by discarding lines and columns after resizing to make sure each dimension is a multiple of patch_size
    :param im_size: The desired image size
    :param patch_size: The desired patch size
    :return: a callable that takes a single argument, a PIL image or a tensor, and normalizes it
    """

    return functools.partial(_normalize, im_size=im_size, patch_size=patch_size)

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
    """
    A workaround to  load .tif files into PIL.Image on a windows machine without it crashing (at least it crashes on my machine)
    This is done by loading the image into an ndarray and then loading that into PIL
    :param im_path: Path to the image
    :return: PIL.Image
    """
    return Image.fromarray(imread(im_path))

@torch.no_grad()
def _distance_matrix(x, y=None, p = 2): #pairwise distance of vectors

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
def _knn_classifier(train_features:torch.Tensor, train_labels, test_features, k:int):

    num_test_images, num_chunks = test_features.shape[0], 100
    imgs_per_chunk = max(num_test_images // num_chunks, num_test_images)
    preds = torch.zeros(test_features.shape[0], device=torch.device('cpu'))

    for idx in range(0, num_test_images, imgs_per_chunk):
        features = test_features[idx:min(idx+imgs_per_chunk, num_test_images), :].cpu()
        dists = _distance_matrix(features, train_features)
        _, voters = torch.topk(dists, k, largest=False)
        votes = torch.gather(train_labels.view(1, -1).expand(features.shape[0], -1),1, voters)
        batch_preds, _ = torch.mode(votes, 1)
        preds[idx : min((idx + imgs_per_chunk), num_test_images)] = batch_preds

    return preds

@torch.no_grad()
def _similarity_knn_classifier(train_features:torch.Tensor, train_labels:torch.Tensor, test_features:torch.Tensor, k:int):

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
def _weighted_similarity_knn_classifier(train_features, train_labels, test_features, k, T, num_classes=9):

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
    def __init__(self, data, lbl, k=20, mode='cos', device=None):
        """
        A simple KNN classifier, uses the same device as the "data" and "lbl" tensors by default, can be overridden by
        passing the "device" argument
        :param data: The training data features tesnor (N, M)
        :param lbl: The training data labels tensor (N,)
        :param k: Parameter for the KNN classification, how many points are considered, higher value leads higher bias but lower variance
        :param mode: The KNN calculation mode, 'cos', 'dist', or 'weighted'. 'cos' (cosine similarity) is default and recommended as it's fastest and seems to perform best
        :param device: The device on which to do the classification, if None (default value) uses the device of the data and lbl tensors
        """
        self.data = data if device is None else data.to(device)
        self.labels = lbl if device is None else lbl.to(device)

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
            return _similarity_knn_classifier(self.data, self.labels, x, self.k)
        elif self.mode == 'weighted':
            return _knn_classifier(self.data, self.labels, x, self.k)
        else:
            return _weighted_similarity_knn_classifier(self.data, self.labels, x, self.k)

    def eval(self):
        pass

    def train(self, mode: bool = True):
        pass

    def children(self):
        return []


def load_features(path, device='cpu', load_labels=True, load_coords=False):
    """
    Load features extracted using 'extract_features.py'
    :param path: Path to where the features are stocked
    :param device: Features will be loaded on the selected device
    :param load_labels: whether to load the labels as well, returned labels will be None if False
    :return: (features, labels): The loaded tensors on the desired device
    """
    features = torch.load(os.path.join(path, 'features.pt')).to(device)
    labels = torch.load(os.path.join(path, 'labels.pt')).to(device) if load_labels else None
    if load_coords:
        coords = torch.load(os.path.join(path, 'coords.pt')).to(device)
        return features, labels, coords

    return features, labels


def load_wsi(path_to_wsi, crop_size, patch_size):
    t = normalize_input(im_size=crop_size, patch_size=patch_size)
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

    def __init__(self, in_dim=384, hidden_dims=None, out_dim=9, pretrained_path=None, device='cuda', dropout=None, n_hidden=1):
        super().__init__()
        # self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=False),
        #                          nn.ReLU(),
        #                          nn.Linear(hidden_dim, out_dim, bias=False)).to(device)
        modules = []
        if dropout is not None:
            modules.append(nn.Dropout(dropout))

        if hidden_dims is not None:
            if isinstance(hidden_dims, int):
                hidden_dims = [hidden_dims]*n_hidden
            n_hidden = len(hidden_dims)
            # Add input layer
            modules.extend([nn.Linear(in_dim, hidden_dims[0]),
                            nn.ReLU()])
            if dropout is not None:
                modules.append(nn.Dropout(dropout))

            # Add hidden layers
            for i in range(n_hidden-1):
                modules.extend([nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                                nn.ReLU()])
                if dropout is not None:
                    modules.append(nn.Dropout(dropout))

            # Output layer
            modules.append(nn.Linear(hidden_dims[-1], out_dim))
        else:
            # Single layer
            modules.append(nn.Linear(in_dim, out_dim))

        self.mlp = nn.Sequential(*modules).to(device)

        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path)
            msg = self.load_state_dict(state_dict)
            self.logger.info(f'Loaded MLP state dict with message {msg}')
        else:
            self.logger.info(f'No pretrained MLP given, loading random weights')

    def forward(self, x):
        return self.mlp(x)

# Build map
def build_prediction_map(
        coords_x: np.ndarray,
        coords_y:  np.ndarray,
        feature:  np.ndarray,
        wsi_dim: Optional[tuple] = None,
        default: Optional[float] = -1.,
) -> np.ndarray:
    """
    Build a prediction map based on x, y coordinate and feature vector. Default values if feature is non existing
    for a certain location is -1.
    Parameters
    ----------
    coords_x: np.ndarray of shape (N,)
        Coordinates of x points.
    coords_y: np.ndarray of shape (N,)
        Coordinates of y points.
    feature: np.ndarray of shape (N, M)
        Feature vector.
    wsi_dim: tuple of int, optional
        Size of the original whole slide. The function add a margin around the map if not null. Default value is None.
    default: float, optional
        Value of the pixel when the feature is not defined.
    Returns
    -------
    map: np.ndarray (W, H, M)
        Feature map. The unaffected points use the default value -1.
    """
    # Compute offset of coordinates in pixel (patch intervals)
    interval_x = np.min(np.unique(coords_x)[1:] - np.unique(coords_x)[:-1])
    interval_y = np.min(np.unique(coords_y)[1:] - np.unique(coords_y)[:-1])

    # Define new coordinates
    if wsi_dim is None:
        offset_x = np.min(coords_x)
        offset_y = np.min(coords_y)
    else:
        offset_x = np.min(coords_x) % interval_x
        offset_y = np.min(coords_y) % interval_y

    coords_x_ = ((coords_x - offset_x) / interval_x).astype(int)
    coords_y_ = ((coords_y - offset_y) / interval_y).astype(int)
    # print(coords_x_.min(), coords_x_.max())
    # print(coords_y_.min(), coords_y_.max())

    # Define size of the feature map
    if wsi_dim is None:
        map = default * np.ones((coords_y_.max() + 1, coords_x_.max() + 1, feature.shape[1]))
    else:
        map = default * np.ones((int(wsi_dim[1] / interval_y), int(wsi_dim[0] / interval_x), feature.shape[1]))

    # Affect values to map
    # print(map.shape)
    map[coords_x_, coords_y_] = feature

    return map


def build_disrete_cmap(name: str, background: Optional[np.ndarray] = None) -> Colormap:
    """
    Build colormap for displaying purpose. Could be one of : 'k19'
    Parameters
    ----------
    name: str
        Name of the colormap to build. Should be one of : 'kather19', 'crctp'
    background: np.ndarray, optional
        Color of th background. THis color will be added as the first item of the colormap. Default value is None.
    Returns
    -------
    cmap: Colormap
        The corresponding colormap
    """

    if name == 'kather19':
        colors = np.array([
            [247, 129, 191],  # Pink - Adipose
            [153, 153, 153],  # Gray - Back
            [255, 255, 51],  # Yellow - Debris
            [152, 78, 160],  # Purple - Lymphocytes
            [255, 127, 0],  # Orange - Mucus
            [23, 190, 192],  # Cyan - Muscle
            [166, 86, 40],  # Brown - Normal mucosa
            [55, 126, 184],  # Blue - Stroma
            [228, 26, 28],  # Red - Tumor
        ]) / 255
    elif name == 'kather19crctp':
        colors = np.array([
            [247, 129, 191],  # Pink - Adipose
            [153, 153, 153],  # Gray - Back
            [255, 255, 51],  # Yellow - Debris
            [152, 78, 160],  # Purple - Lymphocytes
            [255, 127, 0],  # Orange - Mucus
            [23, 190, 192],  # Cyan - Muscle
            [166, 86, 40],  # Brown - Normal mucosa
            [55, 126, 184],  # Blue - Stroma
            [228, 26, 28],  # Red - Tumor
            [77, 167, 77],  # Green - Complex Stroma
        ]) / 255
    elif name == 'embedding':
        colors = np.array([
            [1.0, 0.6, 0.333],  # Orange (src)
            [0.267, 0.667, 0.0],  # Green (target)
        ])
    else:
        # Set of 8 colors (see https://matplotlib.org/stable/tutorials/colors/colormaps.html)
        colors = np.array(cm.get_cmap('Accent').colors)

    if background is not None:
        colors = np.concatenate((background, colors), axis=0)

    cmap = ListedColormap(colors, name='cmap_k19', N=colors.shape[0])
    return cmap

def plot_classification(
        image: Image,
        coords_x: np.ndarray,
        coords_y: np.ndarray,
        cls: np.ndarray,
        cls_labels: np.ndarray,
        wsi_dim: Tuple[int, int],
        save_path: str,
        cmap: Optional[str] = 'kather19',
) -> None:
    """
    Create a plot compose of 2 subplot representing the original image and the classification result.
    Parameters
    ----------
    image: PIL.Image
        Thumbnail of the whole slide image.
    coords_x: np.ndarray of shape (n_samples, )
        x coordinates of the patches in the wsi referential.
    coords_y: np.ndarray of shape (n_samples, )
        y coordinates of the patches in the wsi referential.
    cls: np.ndarray of shape (n_samples, )
        Classes of each classified patch.
    cls_labels: np.ndarray of shape (n_classes, )
        Name of the classes.
    wsi_dim: tuple of int
        Dimension of the original image slide. Should correspond to coord referential.
    save_path: str
        Output path to image to save. Can be JPEG, PNG, PDF.
    cmap: str, optional
        Name of the colomap to use for classification display. Default is `k19`
    """

    # Generate map
    map = build_prediction_map(coords_x, coords_y, cls[:, None], wsi_dim=wsi_dim)[:, :, 0]

    # Create plot dimensions and scaling factor for scatter plot
    fig_size = (int(2 * 12 * image.size[0] / image.size[1]), 12)



    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=fig_size, gridspec_kw={'width_ratios': [1, 1, 0.05]})
    plt.suptitle(os.path.basename(save_path))

    # Plot reference image
    axes[0].set_title('Image thumbnail')
    axes[0].axis('off')
    axes[0].imshow(image)
    # Plot classification map
    axes[1].set_title('Image classification')
    axes[1].axis('off')
    r = axes[1].imshow(
        map + 1,  # Plot map with offset of 1 (avoid background = -1)
        cmap=build_disrete_cmap(cmap, background=[[1.0, 1.0, 1.0]]),  # choose the background color for cls = 0
        interpolation='nearest',  # Avoid interpolation aliasing when scaling image
        vmin=0, vmax=len(cls_labels),
    )
    # axes[1].imsave(save_path)

    # Define color bar with background color
    cls_new_labels = np.concatenate((['-'], cls_labels))
    cax = fig.colorbar(
        r,  # reference map for color
        cax=axes[2],  # axis to use to plot image
        orientation='vertical',  # orientation of the colorbar
        boundaries=np.arange(len(cls_new_labels) + 1) - 0.5  # Span range for color
    )
    cax.set_ticks(np.arange(len(cls_new_labels)))  # Define ticks position (center of colors)
    cax.set_ticklabels(cls_new_labels)  # Define names of labels

    # Save plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
