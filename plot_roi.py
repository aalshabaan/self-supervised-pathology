import numpy as np

import Abed_utils
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
from PIL import Image
from tqdm import tqdm

@torch.inference_mode()
def main():
    outpath = os.path.join(Abed_utils.OUTPUT_ROOT, 'roi_preds', 'ground_truth')
    os.makedirs(outpath, exist_ok=True)
    roi_names = os.listdir(Abed_utils.BERN_TILES_ROOT)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    features, labels = Abed_utils.load_features(os.path.join(Abed_utils.OUTPUT_ROOT, 'features-tuned'), device=device)

    classifier = Abed_utils.KNNClassifier(features, labels, device=device)
    batch_size=64
    for roi in tqdm(roi_names):
        features, lbl, coords = Abed_utils.load_features(os.path.join(Abed_utils.OUTPUT_ROOT, f'features-{roi}'), load_coords=True, device=device)
        coords = coords.cpu().numpy()
        # print(coords.shape)
        # exit(512)
        preds = torch.zeros(features.shape[0]).to(device)
        # for i in range(int(np.ceil(features.shape[0]/batch_size))):
        #     idx = slice(i*batch_size, min((i+1)*batch_size, features.shape[0]))
        #     preds[idx] = classifier(features[idx])

        map = Abed_utils.build_prediction_map(coords[:,1], coords[:, 0], lbl.cpu().numpy()[:, np.newaxis]).squeeze()

        plt.imsave(os.path.join(outpath, f'{roi.split(".")[0]}-label.png'),
                   map+1,
                   # interpolation='nearest',
                   cmap=Abed_utils.build_disrete_cmap('kather19', background=[[1, 1, 1]]),
                   vmin=0, vmax=9,
                   )

        # plt.show()

        # Abed_utils.plot_classification(Image.new('1', (100,100), 1),
        #                                coords[:,1],
        #                                coords[:,0],
        #                                preds.cpu().numpy(),
        #                                ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'],
        #                                wsi_dim=None,
        #                                save_path=os.path.join(Abed_utils.OUTPUT_ROOT, f'roi-{roi}.pdf'),
        #                                cmap='kather19')

if __name__ == '__main__':
    main()