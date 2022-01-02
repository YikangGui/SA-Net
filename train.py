import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from skimage import io, transform
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import StateDataset, Rescale, ToTensor, StateDetect


BATCH_SIZE = 32
EPOCH = 100


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


label_dict = {
    'atHome': 0,
    'onConveyor': 1,
    'inFront': 2,
    'atBin': 3
}


def get_stats(pred, label):
    acc = np.sum(np.array(pred) == np.array(label)) / len(label)
    return acc


class TrainState(object):
    def __init__(self, validation_ratio=0.1, rgb_pwd='data/RealSense/state/rgb', csv_pwd='data/RealSense/state/label.csv'):
        self.validation_ratio = validation_ratio
        self.rgb_pwd = rgb_pwd
        self.csv_pwd = csv_pwd

        train = StateDataset(csv_file='data/RealSense/state/label.csv',
                     root_dir='data/RealSense/state/rgb',
                     transform=transforms.Compose([
                         Rescale(),
                         ToTensor()]))
        train_subset, val_subset = torch.utils.data.random_split(
            train, [5000, 684], generator=torch.Generator().manual_seed(1))
        self.train_loader = DataLoader(dataset=train_subset, shuffle=True, batch_size=BATCH_SIZE, num_workers=8)
        self.val_loader = DataLoader(dataset=val_subset, shuffle=False, batch_size=BATCH_SIZE, num_workers=8)
        self.model = StateDetect().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def main(self):
        print('Start Training...')
        for epoch in range(EPOCH):  # loop over the dataset multiple times
            train_loss_onion = []
            train_loss_eef = []
            train_pred_onion = []
            train_label_onion = []
            train_pred_eef = []
            train_label_eef = []

            self.model.train()
            for i, data in tqdm(enumerate(self.train_loader, 0)):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data['image'].float().to(device)
                onion = data['onion'].to(device)
                eef = data['eef'].to(device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs_onion, outputs_eef = self.model(inputs)
                loss_onion = self.criterion(outputs_onion, onion)
                loss_eef = self.criterion(outputs_eef, eef)
                loss = loss_onion + loss_eef
                loss.backward()
                self.optimizer.step()

                # print statistics
                train_loss_onion.append(loss_onion.item())
                train_loss_eef.append(loss_eef.item())
                train_pred_onion.extend(torch.argmax(outputs_onion, axis=1).cpu().numpy().tolist())
                train_pred_eef.extend(torch.argmax(outputs_eef, axis=1).cpu().numpy().tolist())
                train_label_onion.extend(onion.cpu().numpy().tolist())
                train_label_eef.extend(eef.cpu().numpy().tolist())

            self.model.eval()
            val_loss_onion = []
            val_loss_eef = []
            val_pred_onion = []
            val_label_onion = []
            val_pred_eef = []
            val_label_eef = []
            with torch.no_grad():
                for i, data in enumerate(self.val_loader, 0):
                    inputs = data['image'].float().to(device)
                    onion = data['onion'].to(device)
                    eef = data['eef'].to(device)

                    outputs_onion, outputs_eef = self.model(inputs)
                    loss_onion = self.criterion(outputs_onion, onion)
                    loss_eef = self.criterion(outputs_eef, eef)

                    val_loss_onion.append(loss_onion.item())
                    val_loss_eef.append(loss_eef.item())
                    val_pred_onion.extend(torch.argmax(outputs_onion, axis=1).cpu().numpy().tolist())
                    val_pred_eef.extend(torch.argmax(outputs_eef, axis=1).cpu().numpy().tolist())
                    val_label_onion.extend(onion.cpu().numpy().tolist())
                    val_label_eef.extend(eef.cpu().numpy().tolist())

            print(f'epoch: {epoch} | '
                  f'train onion acc: {round(get_stats(train_pred_onion, train_label_onion) * 100, 2)}% | '
                  f'train eef acc: {round(get_stats(train_pred_eef, train_label_eef) * 100, 2)}% | '
                  f'train onion loss: {round(np.mean(train_loss_onion), 3)} | '
                  f'train eef loss: {round(np.mean(train_loss_eef), 3)} | '
                  f'val onion acc: {round(get_stats(val_pred_onion, val_label_onion) * 100, 2)}% | '
                  f'val eef acc: {round(get_stats(val_pred_eef, val_label_eef) * 100, 2)}% | '                  
                  f'val onion loss: {round(np.mean(val_loss_onion), 3)} | '
                  f'val eef loss: {round(np.mean(val_loss_eef), 3)}')

        print('Finished Training')


if __name__ == '__main__':
    state = TrainState()
    state.main()
