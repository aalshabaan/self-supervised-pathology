import Abed_utils
from segment_wsi import build_disrete_cmap

import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

if __name__ == '__main__':
    fig_name = 'TSNE.pdf'
    tsne_name = 'decomposed.npy'
    out_subdir = 'K_19_tsne'
    os.makedirs(os.path.join(Abed_utils.OUTPUT_ROOT, out_subdir), exist_ok=True)
    features, labels = Abed_utils.load_features(os.path.join(Abed_utils.OUTPUT_ROOT, 'features'))
    if not os.path.exists(os.path.join(Abed_utils.OUTPUT_ROOT, out_subdir, tsne_name)):

        print('Performing TSNE')
        model = make_pipeline(StandardScaler(), PCA(20), TSNE(2), verbose=True)
        t_sne = model.fit_transform(features)

        np.save(file=os.path.join(Abed_utils.OUTPUT_ROOT, out_subdir, tsne_name), arr=t_sne)
    else:
        t_sne = np.load(os.path.join(Abed_utils.OUTPUT_ROOT, out_subdir, tsne_name))

    print('Plotting')
    tags = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
    labels = [tags[x] for x in labels.long()]
    sns.scatterplot(x=t_sne[:, 0], y=t_sne[:, 1], hue=labels, palette=build_disrete_cmap('kather19').colors)

    # plt.scatter(x=t_sne[:, 0], y=t_sne[:, 1], c=labels.long().numpy(), cmap='tab10')
    plt.title('T-SNE Decomposition of Kather-19 DINO Features')
    plt.axis(False)
    plt.tight_layout()
    plt.legend()
    print(f'Saving to {fig_name}')
    plt.savefig(fig_name)
    plt.show()