import functools
import logging
import math
import os
from datetime import datetime

import Abed_utils

import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader, TensorDataset
from tqdm import trange

if __name__ == '__main__':
    os.makedirs('./logs/', exist_ok=True)
    logfile = f'./logs/200_100_hidden.txt'
    weights_file = 'classifier_K19_CE_100ep_200_100_hidden.pt'

    outpath = os.path.join(Abed_utils.OUTPUT_ROOT, 'classifier_weights')
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(os.path.join(Abed_utils.OUTPUT_ROOT, 'classifier_hist'), exist_ok=True)

    logging.basicConfig(filename=logfile,
                        filemode='a',
                        datefmt=Abed_utils.DATETIME_FORMAT,
                        level=logging.DEBUG)
    logger = logging.getLogger('Training')

    patch_size = 8
    lr = 1e-3
    epochs = 100
    batch_size = 64

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.debug(f'using {device}')

    # backbone = Abed_utils.get_model(patch_size, './ckpts/dino_deitsmall8_pretrain.pth').to(device)

    # t = functools.partial(Abed_utils.normalize_input, im_size=224, patch_size=patch_size)
    # ds = ImageFolder(Abed_utils.DATA_ROOT, transform=t, loader=Abed_utils.load_tif_windows)
    features, labels = Abed_utils.load_features(os.path.join(Abed_utils.OUTPUT_ROOT, 'features'), cuda=True)
    # features_flipped, labels_flipped = Abed_utils.load_features(os.path.join(Abed_utils.OUTPUT_ROOT, 'features_flipped'), cuda=True)
    #
    # features = torch.concat([features, features_flipped], 0)
    # labels = torch.concat([labels, labels_flipped])

    train_idx, test_idx = random_split(range(labels.shape[0]),
                                       [9*labels.shape[0]//10,
                                       labels.shape[0] - 9*labels.shape[0]//10])

    X_train, y_train = features[train_idx,:], labels[train_idx]
    X_test, y_test = features[test_idx,:], labels[test_idx]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=True)

    model = Abed_utils.ClassificationHead(dropout=0.2, hidden_dims=[200,100])

    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    for epoch in trange(epochs, desc='Training classifier...'):
        train_loss = 0
        train_correct = 0
        test_loss = 0
        test_correct = 0
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            preds = model(x.to(device))
            loss = criterion(preds, y.to(device))
            loss.backward()

            with torch.no_grad():
                train_correct += torch.argmax(preds, dim=1).eq(y).byte().sum().item()
                train_loss += loss.item()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            for x_t, y_t in test_loader:
                preds = model(x_t.to(device))
                loss = criterion(preds, y_t.to(device))
                test_correct += torch.argmax(preds, dim=1).eq(y_t).byte().sum().item()
                test_loss += loss.item()

        train_loss /= len(train_loader)
        train_acc = train_correct / len(train_loader.dataset)
        test_loss /= len(test_loader)
        test_acc = test_correct / len(test_loader.dataset)

        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)

        logger.info(f'epoch {epoch+1}, train_loss:{train_loss}, train_acc:{train_acc}, val_loss:{test_loss}, val_acc:{test_acc} ')

        # torch.save(model.state_dict(), os.path.join(Abed_utils.OUTPUT_ROOT, 'classifier_weights', f'ckpt{epoch}.pt'))

    data = {'train_loss': train_loss_hist,
            'train_acc': train_acc_hist,
            'test_loss': test_loss_hist,
            'test_acc': test_acc_hist}

    weights_file = os.path.join(os.getcwd(), 'ckpts', weights_file)
    logger.info(f'Saving to {weights_file}')
    torch.save(data, os.path.join(Abed_utils.OUTPUT_ROOT, 'classifier_hist', weights_file))
    torch.save(model.state_dict(), weights_file)