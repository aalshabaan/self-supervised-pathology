import os.path

import Abed_utils
from argparse import ArgumentParser
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn


def fine_tune_model(model:nn.Module, data:DataLoader, lr:float, epochs:int, save_path:str,  layers_to_freeze:int=9, save_freq=5):
    os.makedirs(save_path, exist_ok=True)

    model.train()
    loss = nn.CrossEntropyLoss()
    list(model.children())[0].blocks[layers_to_freeze-1].requires_grad_(False)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epochs):
        for batch, lbl in tqdm(data, f'epoch {e+1}/{epochs}'):
            optim.zero_grad()
            out = model(batch)
            batch_loss = loss(out.softmax(dim=1), lbl)
            batch_loss.backward()
        if not e % save_freq:
            torch.save(model.state_dict(), os.path.join(save_path, f'finetuned{e+1}.pt'))
            print(f'Saved after {e+1} epochs!')

    torch.save(model.state_dict(), os.path.join(save_path, 'finetuned_model.pt'))


def main(args):
    device = torch.device(f'cuda:{args.cuda_dev}' if torch.cuda.is_available() and args.cuda_dev else 'cpu')
    backbone = Abed_utils.get_vit(args.patch_size, args.weight_path, args.key, device, args.arch)
    classifier = Abed_utils.ClassificationHead(device=device)
    model = nn.Sequential(backbone, classifier)
    dataset = ImageFolder(Abed_utils.K_19_PATH, transform=Abed_utils.normalize_input(224, 8), loader=Abed_utils.load_tif_windows)
    data = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=True, drop_last=False)

    fine_tune_model(model,
                    data,
                    lr=args.lr,
                    epochs=args.epochs,
                    save_path=os.path.join(Abed_utils.OUTPUT_ROOT, 'tuned_supervised'),
                    save_freq=args.save_freq)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--cuda-dev', default=None)
    parser.add_argument('--patch-size', default=8, type=int)
    parser.add_argument('--arch', default='vit_small')
    parser.add_argument('--key', default=None)
    parser.add_argument('--weight-path', default=None)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--save-freq', default=5, type=int)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)