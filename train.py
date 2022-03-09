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
from model import *
import random
import time
from datetime import datetime

BATCH_SIZE_STATE = 16
BATCH_SIZE_ACTION = 32
EPOCH = 100
EPOCH_START = 0
PRELOAD_IMAGE = False

use_cuda = torch.cuda.is_available()
cuda_id = 0
device = torch.device(f"cuda:{cuda_id}" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
# torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

label_dict = {
    'atHome': 0,
    'onConveyor': 1,
    'inFront': 2,
    'atBin': 3
}


def get_stats(pred, label):
    acc = np.sum(np.array(pred) == np.array(label)) / len(label)
    return acc


def load_image(dir):
    images = os.listdir(dir)
    image_pool = {}
    for img in tqdm(images):
        image_pool[img] = io.imread(f'{dir}/{img}')
    return image_pool


class TrainState(object):
    def __init__(self, validation_ratio=0.1, rgb_pwd='data/RealSense/train/rgb',
                 csv_pwd='data/RealSense/train/label.csv'):
        self.validation_ratio = validation_ratio

        train = StateDataset(csv_file='data/RealSense/train/train_label.csv',
                             rgb_dir='data/RealSense/train/rgb',
                             transform=transforms.Compose([
                                 # Rescale(),
                                 ToTensor()]))
        val = StateDataset(csv_file='data/RealSense/val/val_label.csv',
                           rgb_dir='data/RealSense/val/rgb',
                           transform=transforms.Compose([
                               # Rescale(),
                               ToTensor()]))
        # test = StateDataset(csv_file='data/RealSense/test/test_label.csv',
        #                    rgb_dir='data/RealSense/test/rgb',
        #                    transform=transforms.Compose([
        #                        # Rescale(),
        #                        ToTensor()]))
        # train_subset, val_subset = torch.utils.data.random_split(
        #     train, [5000, 684], generator=torch.Generator().manual_seed(1))
        self.train_loader = DataLoader(dataset=train, shuffle=True, batch_size=BATCH_SIZE_STATE, num_workers=8,
                                       pin_memory=True, worker_init_fn=seed_worker, generator=g)
        self.val_loader = DataLoader(dataset=val, shuffle=False, batch_size=BATCH_SIZE_STATE, num_workers=8,
                                     pin_memory=True, worker_init_fn=seed_worker, generator=g)
        # self.test_loader = DataLoader(dataset=test, shuffle=False, batch_size=BATCH_SIZE_STATE, num_workers=8,
        #                              pin_memory=True, worker_init_fn=seed_worker, generator=g)
        self.model = StateDetect().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4, weight_decay=1e-4)

    def main(self, weights=None):
        print('Start Training...')
        if weights:
            print('loading weights...')
            self.model.load_state_dict(torch.load(f'./model/state/{weights}.pth'))
            print('loaded weights...')
            EPOCH = weights + 100
            EPOCH_START = weights + 1

        for epoch in range(EPOCH_START, EPOCH):  # loop over the dataset multiple times
            train_loss_onion = []
            train_loss_eef = []
            train_pred_onion = []
            train_label_onion = []
            train_pred_eef = []
            train_label_eef = []
            start_time = time.time()

            self.model.train()
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data['image'].float().to(device)
                onion = data['onion'].to(device)
                eef = data['eef'].to(device)
                # name = data['name']

                # forward + backward + optimize
                time2 = time.time()
                with torch.cuda.amp.autocast():
                    outputs_onion, outputs_eef = self.model(inputs)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                loss_onion = self.criterion(outputs_onion, onion)
                loss_eef = self.criterion(outputs_eef, eef)
                loss = loss_onion + loss_eef
                # print(loss)
                loss.backward()
                self.optimizer.step()
                time3 = time.time()

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

            val_result = pd.DataFrame(columns=['name', 'pred_onion', 'real_onion', 'pred_eef', 'real_eef'])
            with torch.no_grad():
                for i, data in enumerate(self.val_loader, 0):
                    inputs = data['image'].float().to(device)
                    onion = data['onion'].to(device)
                    eef = data['eef'].to(device)
                    name = data['name']

                    with torch.cuda.amp.autocast():
                        outputs_onion, outputs_eef = self.model(inputs)
                    loss_onion = self.criterion(outputs_onion, onion)
                    loss_eef = self.criterion(outputs_eef, eef)

                    val_loss_onion.append(loss_onion.item())
                    val_loss_eef.append(loss_eef.item())
                    val_pred_onion.extend(torch.argmax(outputs_onion, axis=1).cpu().numpy().tolist())
                    val_pred_eef.extend(torch.argmax(outputs_eef, axis=1).cpu().numpy().tolist())
                    val_label_onion.extend(onion.cpu().numpy().tolist())
                    val_label_eef.extend(eef.cpu().numpy().tolist())

                    df = pd.DataFrame(list(zip(name,
                                               torch.argmax(outputs_onion, axis=1).cpu().numpy(),
                                               onion.cpu().numpy(),
                                               torch.argmax(outputs_eef, axis=1).cpu().numpy(),
                                               eef.cpu().numpy())),
                                      columns=['name', 'pred_onion', 'real_onion', 'pred_eef', 'real_eef'])
                    val_result = pd.concat([val_result, df])

            print(f'epoch: {epoch} | '
                  f'train onion acc: {round(get_stats(train_pred_onion, train_label_onion) * 100, 2)}% | '
                  f'train eef acc: {round(get_stats(train_pred_eef, train_label_eef) * 100, 2)}% | '
                  f'train onion loss: {round(np.mean(train_loss_onion), 3)} | '
                  f'train eef loss: {round(np.mean(train_loss_eef), 3)} | '
                  f'val onion acc: {round(get_stats(val_pred_onion, val_label_onion) * 100, 2)}% | '
                  f'val eef acc: {round(get_stats(val_pred_eef, val_label_eef) * 100, 2)}% | '
                  f'val onion loss: {round(np.mean(val_loss_onion), 3)} | '
                  f'val eef loss: {round(np.mean(val_loss_eef), 3)} | '
                  f'collapsed time: {round(time.time() - start_time, 1)}s | '
                  f'current time: {datetime.now().strftime("%H:%M:%S")}')
            torch.save(self.model.state_dict(), f'model/state/{epoch}.pth')
            val_result.to_csv(f'model/state/{epoch}.csv', index=False)
        print('Finished Training')

    def test(self, weights, pwd):
        print('loading weights...')
        self.model.load_state_dict(torch.load(f'{pwd}/{weights}.pth'))
        self.model.eval()
        print('loaded weights...')

        val_result = pd.DataFrame(columns=['name', 'pred_onion', 'real_onion', 'pred_eef', 'real_eef'])

        val_loss_onion = []
        val_loss_eef = []
        val_pred_onion = []
        val_label_onion = []
        val_pred_eef = []
        val_label_eef = []
        val_onion_dist = []
        val_eef_dist = []

        with torch.no_grad():
            start_time = time.time()
            for i, data in enumerate(self.val_loader, 0):
            #  for i, data in enumerate(self.test_loader, 0):
                inputs = data['image'].float().to(device)
                onion = data['onion'].to(device)
                eef = data['eef'].to(device)
                name = data['name']

                with torch.cuda.amp.autocast():
                    outputs_onion, outputs_eef = self.model(inputs)
                loss_onion = self.criterion(outputs_onion, onion)
                loss_eef = self.criterion(outputs_eef, eef)

                val_loss_onion.append(loss_onion.item())
                val_loss_eef.append(loss_eef.item())
                val_pred_onion.extend(torch.argmax(outputs_onion, axis=1).cpu().numpy().tolist())
                val_pred_eef.extend(torch.argmax(outputs_eef, axis=1).cpu().numpy().tolist())
                val_label_onion.extend(onion.cpu().numpy().tolist())
                val_label_eef.extend(eef.cpu().numpy().tolist())
                val_onion_dist.append(F.softmax(outputs_onion))
                val_eef_dist.append(F.softmax(outputs_eef))

                df = pd.DataFrame(list(zip(name,
                                           torch.argmax(outputs_onion, axis=1).cpu().numpy(),
                                           onion.cpu().numpy(),
                                           torch.argmax(outputs_eef, axis=1).cpu().numpy(),
                                           eef.cpu().numpy())),
                                  columns=['name', 'pred_onion', 'real_onion', 'pred_eef', 'real_eef'])
                val_result = pd.concat([val_result, df])

        np.save('./dist/onion_dist.npy', torch.cat(val_onion_dist).cpu().numpy())
        np.save('./dist/eef_dist.npy', torch.cat(val_eef_dist).cpu().numpy())

        print(f'val onion acc: {round(get_stats(val_pred_onion, val_label_onion) * 100, 2)}% | '
              f'val eef acc: {round(get_stats(val_pred_eef, val_label_eef) * 100, 2)}% | '
              f'val onion loss: {round(np.mean(val_loss_onion), 3)} | '
              f'val eef loss: {round(np.mean(val_loss_eef), 3)} | '
              f'collapsed time: {round(time.time() - start_time, 1)}s | '
              f'current time: {datetime.now().strftime("%H:%M:%S")}')


class TrainStateNew(object):
    def __init__(self):
        # self.validation_ratio = validation_ratio

        train = StateDatasetNew(csv_file='data/Data/train.csv',
                             rgb_dir='data/Data/train',
                             transform=transforms.Compose([
                                 # Rescale(),
                                 ToTensor()]))
        val = StateDatasetNew(csv_file='data/Data/val.csv',
                           rgb_dir='data/Data/val',
                           transform=transforms.Compose([
                               # Rescale(),
                               ToTensor()]))
        test = StateDatasetNew(csv_file='data/Data/test.csv',
                           rgb_dir='data/Data/test',
                           transform=transforms.Compose([
                               # Rescale(),
                               ToTensor()]))
        # train_subset, val_subset = torch.utils.data.random_split(
        #     train, [5000, 684], generator=torch.Generator().manual_seed(1))
        self.train_loader = DataLoader(dataset=train, shuffle=True, batch_size=BATCH_SIZE_STATE, num_workers=8,
                                       pin_memory=True, worker_init_fn=seed_worker, generator=g)
        self.val_loader = DataLoader(dataset=val, shuffle=False, batch_size=BATCH_SIZE_STATE, num_workers=8,
                                     pin_memory=True, worker_init_fn=seed_worker, generator=g)
        self.test_loader = DataLoader(dataset=test, shuffle=False, batch_size=BATCH_SIZE_STATE, num_workers=8,
                                     pin_memory=True, worker_init_fn=seed_worker, generator=g)
        self.model = StateDetectNew().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4, weight_decay=1e-4)

    def main(self, weights=None):
        print('Start Training...')
        global EPOCH_START, EPOCH
        if weights:
            print('loading weights...')
            self.model.load_state_dict(torch.load(f'./model/state/{weights}.pth'))
            print('loaded weights...')
            EPOCH = weights + 100
            EPOCH_START = weights + 1

        for epoch in range(EPOCH_START, EPOCH):  # loop over the dataset multiple times
            train_loss_grasp = []
            train_loss_eef = []
            train_pred_grasp = []
            train_label_grasp = []
            train_pred_eef = []
            train_label_eef = []
            start_time = time.time()

            self.model.train()
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data['image'].float().to(device)
                grasp = data['grasp'].to(device)
                eef = data['eef'].to(device)
                # name = data['name']

                # forward + backward + optimize
                time2 = time.time()
                with torch.cuda.amp.autocast():
                    outputs_eef, outputs_grasp = self.model(inputs)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                loss_grasp = self.criterion(outputs_grasp, grasp)
                loss_eef = self.criterion(outputs_eef, eef)
                loss = loss_grasp + loss_eef
                # print(loss)
                loss.backward()
                self.optimizer.step()
                time3 = time.time()

                # print statistics
                train_loss_grasp.append(loss_grasp.item())
                train_loss_eef.append(loss_eef.item())
                train_pred_grasp.extend(torch.argmax(outputs_grasp, axis=1).cpu().numpy().tolist())
                train_pred_eef.extend(torch.argmax(outputs_eef, axis=1).cpu().numpy().tolist())
                train_label_grasp.extend(grasp.cpu().numpy().tolist())
                train_label_eef.extend(eef.cpu().numpy().tolist())

            self.model.eval()

            val_loss_grasp = []
            val_loss_eef = []
            val_pred_grasp = []
            val_label_grasp = []
            val_pred_eef = []
            val_label_eef = []

            val_result = pd.DataFrame(columns=['name', 'pred_grasp', 'real_grasp', 'pred_eef', 'real_eef'])
            with torch.no_grad():
                for i, data in enumerate(self.val_loader, 0):
                    inputs = data['image'].float().to(device)
                    grasp = data['grasp'].to(device)
                    eef = data['eef'].to(device)
                    name = data['name']

                    with torch.cuda.amp.autocast():
                        outputs_eef, outputs_grasp = self.model(inputs)
                    loss_grasp = self.criterion(outputs_grasp, grasp)
                    loss_eef = self.criterion(outputs_eef, eef)

                    val_loss_grasp.append(loss_grasp.item())
                    val_loss_eef.append(loss_eef.item())
                    val_pred_grasp.extend(torch.argmax(outputs_grasp, axis=1).cpu().numpy().tolist())
                    val_pred_eef.extend(torch.argmax(outputs_eef, axis=1).cpu().numpy().tolist())
                    val_label_grasp.extend(grasp.cpu().numpy().tolist())
                    val_label_eef.extend(eef.cpu().numpy().tolist())

                    df = pd.DataFrame(list(zip(name,
                                               torch.argmax(outputs_grasp, axis=1).cpu().numpy(),
                                               grasp.cpu().numpy(),
                                               torch.argmax(outputs_eef, axis=1).cpu().numpy(),
                                               eef.cpu().numpy())),
                                      columns=['name', 'pred_grasp', 'real_grasp', 'pred_eef', 'real_eef'])
                    val_result = pd.concat([val_result, df])

            print(f'epoch: {epoch} | '
                  f'train grasp acc: {round(get_stats(train_pred_grasp, train_label_grasp) * 100, 2)}% | '
                  f'train eef acc: {round(get_stats(train_pred_eef, train_label_eef) * 100, 2)}% | '
                  f'train grasp loss: {round(np.mean(train_loss_grasp), 3)} | '
                  f'train eef loss: {round(np.mean(train_loss_eef), 3)} | '
                  f'val grasp acc: {round(get_stats(val_pred_grasp, val_label_grasp) * 100, 2)}% | '
                  f'val eef acc: {round(get_stats(val_pred_eef, val_label_eef) * 100, 2)}% | '
                  f'val grasp loss: {round(np.mean(val_loss_grasp), 3)} | '
                  f'val eef loss: {round(np.mean(val_loss_eef), 3)} | '
                  f'collapsed time: {round(time.time() - start_time, 1)}s | '
                  f'current time: {datetime.now().strftime("%H:%M:%S")}')
            torch.save(self.model.state_dict(), f'model/state/{epoch}.pth')
            val_result.to_csv(f'model/state/{epoch}.csv', index=False)
        print('Finished Training')

    def test(self, weights, pwd):
        print('loading weights...')
        self.model.load_state_dict(torch.load(f'{pwd}/{weights}.pth', map_location=f"cuda:{cuda_id}"))
        self.model.eval()
        print('loaded weights...')

        test_result = pd.DataFrame(columns=['name', 'pred_grasp', 'real_grasp', 'pred_eef', 'real_eef'])

        test_loss_grasp = []
        test_loss_eef = []
        test_pred_grasp = []
        test_label_grasp = []
        test_pred_eef = []
        test_label_eef = []
        test_grasp_dist = []
        test_eef_dist = []

        with torch.no_grad():
            start_time = time.time()
            for i, data in enumerate(self.test_loader, 0):
                inputs = data['image'].float().to(device)
                grasp = data['grasp'].to(device)
                eef = data['eef'].to(device)
                name = data['name']

                # print(torch.cuda.memory_summary())
                outputs_eef, outputs_grasp = self.model(inputs)
                loss_grasp = self.criterion(outputs_grasp, grasp)
                loss_eef = self.criterion(outputs_eef, eef)

                # print(loss_grasp, name)
                test_loss_grasp.append(loss_grasp.item())
                test_loss_eef.append(loss_eef.item())
                test_pred_grasp.extend(torch.argmax(outputs_grasp, axis=1).cpu().numpy().tolist())
                test_pred_eef.extend(torch.argmax(outputs_eef, axis=1).cpu().numpy().tolist())
                test_label_grasp.extend(grasp.cpu().numpy().tolist())
                test_label_eef.extend(eef.cpu().numpy().tolist())
                test_grasp_dist.append(F.softmax(outputs_grasp))
                test_eef_dist.append(F.softmax(outputs_eef))

                df = pd.DataFrame(list(zip(name,
                                           torch.argmax(outputs_grasp, axis=1).cpu().numpy(),
                                           grasp.cpu().numpy(),
                                           torch.argmax(outputs_eef, axis=1).cpu().numpy(),
                                           eef.cpu().numpy())),
                                  columns=['name', 'pred_grasp', 'real_grasp', 'pred_eef', 'real_eef'])
                test_result = pd.concat([test_result, df])

        np.save('./dist/grasp_dist.npy', torch.cat(test_grasp_dist).cpu().numpy())
        np.save('./dist/eef_dist.npy', torch.cat(test_eef_dist).cpu().numpy())

        test_result.to_csv(f'test_result.csv', index=False)

        print(f'test grasp acc: {round(get_stats(test_pred_grasp, test_label_grasp) * 100, 2)}% | '
              f'test eef acc: {round(get_stats(test_pred_eef, test_label_eef) * 100, 2)}% | '
              f'test grasp loss: {round(np.mean(test_loss_grasp), 3)} | '
              f'test eef loss: {round(np.mean(test_loss_eef), 3)} | '
              f'collapsed time: {round(time.time() - start_time, 1)}s | '
              f'current time: {datetime.now().strftime("%H:%M:%S")}')


class TrainAction(object):
    def __init__(self, validation_ratio=0.1, rgb_pwd='data/RealSense/state/rgb',
                 csv_pwd='data/RealSense/state/label.csv'):
        self.validation_ratio = validation_ratio
        self.rgb_pwd = rgb_pwd
        self.csv_pwd = csv_pwd
        self.image_pool = None
        if PRELOAD_IMAGE:
            self.image_pool = load_image(self.rgb_pwd)

        train = ActionDataset(
            transform=transforms.Compose([
                RescaleAction(),
                ToTensorAction()]),
            image_pool=self.image_pool
        )
        train_subset, val_subset = torch.utils.data.random_split(
            train, [5000, 640], generator=torch.Generator().manual_seed(1))
        self.train_loader = DataLoader(dataset=train_subset, shuffle=True, batch_size=BATCH_SIZE_ACTION, num_workers=4,
                                       pin_memory=True, worker_init_fn=seed_worker, generator=g)
        self.val_loader = DataLoader(dataset=val_subset, shuffle=False, batch_size=BATCH_SIZE_ACTION, num_workers=4,
                                     pin_memory=True, worker_init_fn=seed_worker, generator=g)
        self.model = ActionDetect().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def main(self):
        print('Start Action Recognition Training...')
        for epoch in range(EPOCH):  # loop over the dataset multiple times
            train_loss = []
            train_pred = []
            train_label = []
            start_time = time.time()

            self.model.train()
            for i, data in tqdm(enumerate(self.train_loader, 0)):
                time1 = time.time()
                # get the inputs; data is a list of [inputs, labels]
                inputs = data['image'].float().to(device)
                action = data['action'].to(device)
                # name = data['name']
                time2 = time.time()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, action)
                loss.backward()
                self.optimizer.step()
                time3 = time.time()

                train_loss.append(loss.item())
                train_pred.extend(torch.argmax(outputs, axis=1).cpu().numpy().tolist())
                train_label.extend(action.cpu().numpy().tolist())

            self.model.eval()

            val_loss = []
            val_pred = []
            val_label = []

            val_result = pd.DataFrame(columns=['name', 'pred', 'real'])
            with torch.no_grad():
                for i, data in enumerate(self.val_loader, 0):
                    inputs = data['image'].float().to(device)
                    action = data['action'].to(device)
                    name = data['name']

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, action)

                    val_loss.append(loss.item())
                    val_pred.extend(torch.argmax(outputs, axis=1).cpu().numpy().tolist())
                    val_label.extend(action.cpu().numpy().tolist())

                    df = pd.DataFrame(list(zip(name,
                                               torch.argmax(outputs, axis=1).cpu().numpy(),
                                               action.cpu().numpy()
                                               )),
                                      columns=['name', 'pred', 'real'])
                    val_result = pd.concat([val_result, df])

            print(f'epoch: {epoch} | '
                  f'train acc: {round(get_stats(train_pred, train_label) * 100, 2)}% | '
                  f'train loss: {round(np.mean(train_loss), 3)} | '
                  f'val acc: {round(get_stats(val_pred, val_label) * 100, 2)}% | '
                  f'val loss: {round(np.mean(val_loss), 3)} | '
                  f'collapsed time: {round(time.time() - start_time, 1)}s')
            torch.save(self.model.state_dict(), f'model/action/{epoch}.pth')
            val_result.to_csv(f'model/action/{epoch}.csv', index=False)
        print('Finished Training')


if __name__ == '__main__':
    state_new = TrainStateNew()
    # state_new.main()
    state_new.test(77, './model')

    # state = TrainState()
    # state.main(99)
    # state.test(160, './model/state')
    # action = TrainAction()
    # action.main()
