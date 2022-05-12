import functools
import logging
import os
from typing import Union, Optional, Tuple

import PIL
from matplotlib import cm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import openslide
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, Colormap
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.special import softmax

import Abed_utils


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
    if 'NN' in classifier_type:
        map[coords_x_, coords_y_] = feature
    else:
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



def wsi_tensor_into_image(path_to_tensor:Union[str, torch.Tensor], downsampling=100, cmap='tab10', n_classes=9, class_labels=None):

    # clone to free the original huge tensor for garbage collection
    img = torch.load(path_to_tensor)[::downsampling, ::downsampling].clone().t().unsqueeze(2).repeat(1, 1, 3).numpy()
    # print(img.shape)
    if cmap in plt.colormaps:
        colors = plt.colormaps[cmap].colors
    else:
        print(f'invalid colormap {cmap}, using tab10 instead')
        colors = plt.colormaps['tab10'].colors

    colors = list(colors)
    for i in range(n_classes):
        colors[i] = list(colors[i])
        for j in range(3):
            colors[i][j] = round(colors[i][j]*255)
        colors[i] = tuple(colors[i])

    for i in range(n_classes):
        for c in range(3):
            # print(img.shape, colors.shape)
            img[:, :, c][img[:, :, c] == i] = colors[i][c]

    if class_labels is not None:
        legend = Image.new('RGB', (100, img.shape[0]), (255, 255, 255))
        draw = ImageDraw.ImageDraw(legend)
        font = ImageFont.truetype('arial.ttf', 15)
        for i in range(n_classes):
            if i < len(class_labels):
                lbl = class_labels[i]
            else:
                lbl = str(i)
            # box = Image.new('RGB', (math.ceil(legend.width/2), legend.height//9), colors[i])
            draw.rectangle((legend.width/2, i*legend.height/n_classes, legend.width, (i+1)*legend.height/n_classes), fill=colors[i])
            # legend.paste(box, (i*box.height, legend.width/2))
            draw.text((0, (i+0.5)*legend.height/n_classes), text=lbl, font=font, fill=(0,0,0))
        # img:np.ndarray
        img = np.pad(img, ((0, 0), (0, legend.width), (0, 0)))
        img = Image.fromarray(img)
        img.paste(legend, (img.width-legend.width, 0))
        img = np.array(img)

    outpath, filename = os.path.split(path_to_tensor)
    filename = filename.split(os.extsep)[0]+'.png'
    plt.imsave(os.path.join(outpath, filename), img)

def plot_classification(
        image: PIL.Image,
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


@torch.no_grad()
def segment_wsi_abbet_plot(ds, path_to_model, path_to_embeddings, out_filename, K=20,
                           vit_patch_size=8, batch_size=8, model_key=None, classifier=None, classifier_type='KNN'):
    data = DataLoader(ds, batch_size=batch_size, )

    if classifier is None:
        print('No model passed, Creating model and loading pretrained weights')
        if 'NN' in classifier_type:
            features, labels = Abed_utils.load_features(path_to_embeddings, load_labels=True, cuda=True)
            classifier = Abed_utils.KNNClassifier(features, labels, K)  # uses the device of the feature/label tensors
        elif 'MLP' in classifier_type:
            classifier = Abed_utils.ClassificationHead(pretrained_path='./ckpts/classifier_K19_CE_100ep_one_layer.pt')
        else:
            raise ValueError(f'Unknown classifier model {classifier_type}, use "KNN" or "MLP"')

    backbone = Abed_utils.get_vit(vit_patch_size, path_to_model, key=model_key)

    model = nn.Sequential(backbone, classifier)
    model.eval()

    preds = []
    metadata = []

    os.makedirs(outpath, exist_ok=True)

    if 'thumbnail' not in ds.s.associated_images:
        # Append dummy
        thumbnail = Image.fromarray(np.zeros((ds.level_dimensions[-2][0], ds.level_dimensions[-2][1], 3), dtype=np.uint8))
    else:
        thumbnail = ds.s.associated_images['thumbnail']

    for i, crop_pair in enumerate(tqdm(data, desc='Classifying patches...')):
        for img, metas in zip(*crop_pair):
            [mag, level, tx, ty, cx, cy, bx, by, s_src, s_tar] = metas
            preds.extend(model(img.to(next(model.parameters()).device)).cpu().numpy())
            metadata.extend(
                np.array([mag.numpy(), level.numpy(), tx.numpy(), ty.numpy(), cx.numpy(), cy.numpy(), bx.numpy(),
                          by.numpy(), s_src.numpy(), s_tar.numpy()]).T
            )
    preds = np.array(preds)
    metadata = np.array(metadata)

    data = {'name': os.path.basename(ds.path),
            'path': ds.path,
            'model_path': path_to_model,
            'dataset_name': 'Kather-19',
            'classification': preds,
            'classification_labels': ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'],
            'metadata': metadata,
            'metadata_labels': ['mag', 'level', 'tx', 'ty', 'cx', 'cy', 'bx', 'by', 's_src', 's_tar'],
            'thumbnail': thumbnail}

    np.save(os.path.join(outpath, out_filename), arr=data)


@torch.no_grad()
def segment_wsi(wsi, path_to_model, path_to_embeddings, out_filename, K=20,
                vit_patch_size=8, batch_size=8, model_key=None):

    data = DataLoader(wsi, batch_size=batch_size)

    print('Loading Kather-19 embeddings')
    features, labels = Abed_utils.load_features(path_to_embeddings, load_labels=True, cuda=True)

    print('Creating model and loading pretrained weights')
    classifier = Abed_utils.KNNClassifier(features, labels, K)  # uses the device of the feature/label tensors

    backbone = Abed_utils.get_vit(vit_patch_size, path_to_model, key=model_key)
    backbone.eval()

    model = nn.Sequential(backbone, classifier)

    canvas = torch.ones(*wsi.s.dimensions, dtype=torch.uint8) * 255
    print(f'Created canvas of size {canvas.size()}')

    outpath = os.path.join(Abed_utils.OUTPUT_ROOT, 'wsi', os.path.split(Abed_utils.TEST_SLIDE_PATH)[1])
    os.makedirs(outpath, exist_ok=True)

    for i, crop_pair in enumerate(tqdm(data)):
        for img, metas in zip(*crop_pair):
            tx, ty, bx, by = metas[2].long(), metas[3].long(), metas[6].long(), metas[7].long()
            pred = model(img.to(next(model.parameters()).device))
            # print(tx.shape, ty.shape, bx.shape, by.shape, pred.shape)
            for i in range(pred.shape[0]):
                canvas[tx[i]:bx[i], ty[i]:by[i]] = pred[i]

        # if i % 100 == 0:
        #     print('saving')
        #     torch.save(canvas, os.path.join(outpath, 'ckpt.pt'))

    torch.save(canvas, os.path.join(outpath, out_filename))
    return outpath

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    feat_path = os.path.join(Abed_utils.OUTPUT_ROOT, 'features')
    classifier_type = "KNN"
    outpath = os.path.join(Abed_utils.OUTPUT_ROOT, f'predictions_{classifier_type}')
    features, labels = Abed_utils.load_features(feat_path, cuda=True)
    classifier = Abed_utils.KNNClassifier(features, labels)

    wsi_paths = [x for x in os.listdir(Abed_utils.BERN_COHORT_ROOT)
                 if os.path.isdir(os.path.join(Abed_utils.BERN_COHORT_ROOT, x))]

    for path in wsi_paths:
        cur_dir = os.path.join(Abed_utils.BERN_COHORT_ROOT, path)
        wsis = [x for x in os.listdir(cur_dir) if os.path.splitext(x)[1] == '.mrxs']

        for wsi in wsis:
            ds = Abed_utils.load_wsi(os.path.join(cur_dir, wsi), 224, 8)
            out_filename = f'{wsi}_seg_dino_imagenet_100ep_{classifier_type}'
    # ds = Abed_utils.load_wsi(Abed_utils.TEST_SLIDE_PATH, 224, 8)

            if not os.path.exists(os.path.join(outpath, out_filename)):
                segment_wsi_abbet_plot(ds,
                                       './ckpts/dino_deitsmall8_pretrain.pth',
                                       feat_path,
                                       out_filename,
                                       classifier=classifier,
                                       batch_size=8 if Abed_utils.my_pc() else 256)
                #
                logger.info(f'Saved to {os.path.join(outpath, out_filename)}')
            else:
                logger.info(f'Skipped {wsi}, already computed.')
    # outpath = r'D:\self_supervised_pathology\output\wsi\001b_B2005.30530_C_HE.mrxs'
    # logger.debug('Loading classification results')
    # data = np.load(os.path.join(outpath, out_filename), allow_pickle=True).item()
    #
    # logger.debug('Outputting image')
    # plot_classification(data['thumbnail'],
    #                     data['metadata'][:, 3],
    #                     data['metadata'][:, 4],
    #                     data['classification'] if 'NN' in classifier_type else np.argmax(data['classification'], axis=1),
    #                     data['classification_labels'],
    #                     ds.s.dimensions,
    #                     os.path.join(outpath, f'mask_{classifier_type}.png'))

    # logger.debug("Generate detections for QuPath viz ...")
    # # Correction from metadata offset
    # offset_x = int(ds.s.properties.get(openslide.PROPERTY_NAME_BOUNDS_X, 0))
    # offset_y = int(ds.s.properties.get(openslide.PROPERTY_NAME_BOUNDS_Y, 0))
    # # Correction for overlapping tiles
    # # centering = 0.5 * config['wsi']['padding_factor'] * data['metadata'][0, -2]
    # centering = 0.5 * ds.padding_factor * data['metadata'][0, -2]
    # # dataset_name = data.get('dataset_name', config['dataset']['name'])
    # dataset_name = data['dataset_name'] if 'dataset_name' in data else 'Kather-19'
    # # img_path = os.path.join(outpath)
    # # Write classification output overlay for QuPath
    # Abed_utils.save_annotation_qupath(
    #     tx=data['metadata'][:, 2] - offset_x + centering,
    #     ty=data['metadata'][:, 3] - offset_y + centering,
    #     bx=data['metadata'][:, 6] - offset_x - centering,
    #     by=data['metadata'][:, 7] - offset_y - centering,
    #     values=data['classification'] if 'NN' in classifier_type else np.argmax(data['classification'], axis=1),
    #     values_name={k: data['classification_labels'][k] for k in range(len(data['classification_labels']))},
    #     outpath=os.path.join(outpath, f"mask_detection_{classifier_type}.json"),
    #     cmap=build_disrete_cmap(dataset_name),
    # )
    #
    # if 'MLP' in classifier_type:
    #     for i, cls in enumerate(tqdm(data['classification_labels'], desc='Classes predictions ...')):
    #         values = np.clip(softmax(data['classification'], axis=1)[:, i], a_min=0, a_max=0.99)
    #         bins = np.linspace(0, 1, 101)
    #         Abed_utils.save_annotation_qupath(
    #             tx=data['metadata'][:, 2] - offset_x + centering,
    #             ty=data['metadata'][:, 3] - offset_y + centering,
    #             bx=data['metadata'][:, 6] - offset_x - centering,
    #             by=data['metadata'][:, 7] - offset_y - centering,
    #             values=bins[np.digitize(values, bins)],
    #             values_name=bins[np.digitize(values, bins)],
    #             outpath=os.path.join(outpath, "mask_{}_{}_detection.json".format(classifier_type, cls)),
    #             cmap=cm.get_cmap('inferno'),
    #         )
    # # wsi_tensor_into_image(os.path.join(outpath, out_filename), class_labels=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'])
    #
    #
    #
