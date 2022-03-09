import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import pandas as pd
import os
from skimage import io, transform
import matplotlib.pyplot as plt
import torch.optim as optim

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

label_dict = {
    'atHome': 0,
    'onConveyor': 1,
    'inFront': 2,
    'atBin': 3
}

label_dict_new = {
    'on': 0,
    'ins': 1,
    'at': 2
}

grasp_dict = {
    'n': 0,
    'y': 1
}

action_dict = {
    'Claim': 0,
    'Pick': 1,
    'Inspect': 2,
    'PlaceOnConveyor': 3,
    'PlaceOnBin': 4
}


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, stride, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.stride = stride
        # self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.padding = 0, 0
        self.bias = bias

        self.conv_input = nn.Conv2d(in_channels=self.input_dim,
                                    out_channels=4 * self.hidden_dim,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_h = nn.Conv2d(in_channels=self.hidden_dim,
                                out_channels=4 * self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding='same',
                                bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        input_conv = self.conv_input(input_tensor)
        h_conv = self.conv_h(h_cur)
        combined_conv = input_conv + h_conv
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        output_height = int((height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0] + 1)
        output_width = int((width + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1] + 1)

        return (
            torch.zeros(batch_size, self.hidden_dim, output_height, output_width, device=self.conv_input.weight.device),
            torch.zeros(batch_size, self.hidden_dim, output_height, output_width, device=self.conv_input.weight.device),
            output_height, output_width)


class ConvLSTM(nn.Module):
    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
    Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, stride, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          stride=self.stride[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            h, c, height, width = self.cell_list[i].init_hidden(batch_size, image_size)
            image_size = (height, width)
            init_states.append((h, c))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ActionDetect(nn.Module):
    def __init__(self, sequence_size=3):
        super(ActionDetect, self).__init__()
        self.sequence_size = sequence_size
        # TODO added depth image as 4th channel
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2))
        # TODO regularization for kernel, torch.norm(model.layer.weight, p=2)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=(2, 2), stride=(2, 2))
        self.pooling1 = nn.AvgPool2d((2, 2), (1, 1))
        self.convlstm = ConvLSTM(input_dim=16,
                                 hidden_dim=[20, 5],
                                 kernel_size=[(3, 3), (2, 2)],
                                 stride=[(2, 2), (3, 3)],
                                 num_layers=2,
                                 batch_first=False,
                                 bias=True,
                                 return_all_layers=False)
        self.fc = nn.Linear(2470, 5)

    def forward(self, inputs, state_tensor=None):
        # TODO check the size of inputs, currently assume (batch_size, sequence_size, C, H, W)
        conv_outputs = []
        for i in range(self.sequence_size):
            output1 = F.relu(self.conv1(inputs[:, i, :, :, :]))
            output2 = F.relu(self.conv2(output1))
            conv_outputs.append(self.pooling1(output2))
        conv_outputs = torch.stack(conv_outputs)
        _, convlstm_outputs = self.convlstm(conv_outputs)
        convlstm_h, convlstm_c = convlstm_outputs[0]
        # convlstm_outputs = torch.cat([convlstm_outputs, state_tensor], dim=1)
        convlstm_h = torch.flatten(convlstm_h, start_dim=1)
        outputs = self.fc(convlstm_h)
        # outputs = F.softmax(self.fc(convlstm_h), dim=1)
        return outputs


class StateDetect(nn.Module):
    def __init__(self):
        super(StateDetect, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 9), stride=1)
        self.pool1 = nn.MaxPool2d(4, 3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 9), stride=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 9), stride=1)
        self.pool2 = nn.MaxPool2d(2, 3)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 9), stride=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=(3, 9), stride=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # self.activation = nn.

        # self.fc1_1 = nn.Linear((32 * 24 * 24),
        #                        128)  # for input shape: [1, 3, 480, 640], the output shape is: [1, 32, 24, 24]
        self.fc1_1 = nn.Linear((32 * 37 * 60),
                               128)  # for input shape: [1, 3, 480, 640], the output shape is: [1, 32, 24, 24]
        self.fc1_2 = nn.Linear(128, 64)
        self.fc1_3 = nn.Linear(64, 32)
        self.fc1_4 = nn.Linear(32, 4)  # for ee_loc
        self.fc1_5 = nn.Linear(32, 4)  # for o_loc

        self.dropout = nn.Dropout(p=0.75)

        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.kaiming_normal_(self.fc1_1.weight)
        nn.init.kaiming_normal_(self.fc1_2.weight)
        nn.init.kaiming_normal_(self.fc1_3.weight)
        nn.init.kaiming_normal_(self.fc1_4.weight)
        nn.init.kaiming_normal_(self.fc1_5.weight)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = self.pool2(x)
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = self.pool3(x)  # (batch size, 32, 24, 24)

        # x = x.view(-1, 32 * 24 * 24)  # Flatten layer
        x = x.view(-1, 32 * 37 * 60)  # Flatten layer
        # state_info = x.clone()
        x = self.dropout(F.leaky_relu(self.fc1_1(x)))
        x = self.dropout(F.leaky_relu(self.fc1_2(x)))
        x = self.dropout(F.leaky_relu(self.fc1_3(x)))
        x1 = self.fc1_4(x)  # for ee_loc
        x2 = self.fc1_5(x)  # for o_loc
        # x1 = F.softmax(x1, dim=1)
        # x2 = F.softmax(x2, dim=1)
        return x1, x2


class StateDetectNew(nn.Module):
    def __init__(self):
        super(StateDetectNew, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 9), stride=1)
        self.pool1 = nn.MaxPool2d(4, 3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 9), stride=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 9), stride=1)
        self.pool2 = nn.MaxPool2d(2, 3)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 9), stride=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=(3, 9), stride=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # self.activation = nn.

        # self.fc1_1 = nn.Linear((32 * 24 * 24),
        #                        128)  # for input shape: [1, 3, 480, 640], the output shape is: [1, 32, 24, 24]
        self.fc1_1 = nn.Linear((32 * 37 * 60),
                               128)  # for input shape: [1, 3, 480, 640], the output shape is: [1, 32, 24, 24]
        self.fc1_2 = nn.Linear(128, 64)
        self.fc1_3 = nn.Linear(64, 32)
        self.fc1_4 = nn.Linear(32, 3)  # for ee_loc
        self.fc1_5 = nn.Linear(32, 2)  # for onion grasp or not

        self.dropout = nn.Dropout(p=0.75)

        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.kaiming_normal_(self.fc1_1.weight)
        nn.init.kaiming_normal_(self.fc1_2.weight)
        nn.init.kaiming_normal_(self.fc1_3.weight)
        nn.init.kaiming_normal_(self.fc1_4.weight)
        nn.init.kaiming_normal_(self.fc1_5.weight)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = self.pool2(x)
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = self.pool3(x)  # (batch size, 32, 24, 24)

        # x = x.view(-1, 32 * 24 * 24)  # Flatten layer
        x = x.view(-1, 32 * 37 * 60)  # Flatten layer
        # state_info = x.clone()
        x = self.dropout(F.leaky_relu(self.fc1_1(x)))
        x = self.dropout(F.leaky_relu(self.fc1_2(x)))
        x = self.dropout(F.leaky_relu(self.fc1_3(x)))
        x1 = self.fc1_4(x)  # for ee_loc
        x2 = self.fc1_5(x)  # for grasp
        # x1 = F.softmax(x1, dim=1)
        # x2 = F.softmax(x2, dim=1)
        return x1, x2


class ActionDataset(Dataset):
    def __init__(self, action_csv_file='data/RealSense/state/action_label.csv', rgb_dir='data/RealSense/state/rgb',
                 depth_dir='data/RealSense/state/depth', transform=None, image_pool=None):
        self.action_csv_file = action_csv_file
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.image_pool = image_pool
        try:
            self.action_labels = pd.read_csv(action_csv_file, header=None)
        except FileNotFoundError:
            self.action_labels = self.create_action_csv()

    def __len__(self):
        return len(self.action_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.action_labels.iloc[idx]
        images = data[:-1]
        action = data[-1]

        data = []
        for rgb_img in images:
            if self.image_pool:
                rgb = self.image_pool[rgb_img]
            else:
                rgb_img_name = os.path.join(self.rgb_dir, rgb_img)
                rgb = io.imread(rgb_img_name)

            data.append(rgb.copy())
        data = np.array(data)

        sample = {'image': data / 255, 'action': np.array(action_dict[action]), 'name': rgb_img}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def create_action_csv(self):
        index = 0
        action_labels = pd.DataFrame(columns=['image-5', 'image-4', 'image-3', 'image-2', 'image-1', 'action'])
        for i in range(len(self.labels)):
            if pd.isna(self.labels.iloc[i]['action']):
                continue
            action_labels.loc[index] = [self.labels.iloc[j]['image'] for j in range(i - 4, i + 1)] + [
                self.labels.iloc[i]['action']]
            index += 1
        action_labels.to_csv(self.action_csv_file, index=False)
        return action_labels


class StateDataset(Dataset):
    def __init__(self, csv_file, rgb_dir, transform=None, image_pool=None):
        self.labels = pd.read_csv(csv_file, header=None)
        self.rgb_dir = rgb_dir
        self.transform = transform
        self.image_pool = image_pool

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        name, o_loc, e_loc = self.labels.iloc[idx]
        img_name = os.path.join(self.rgb_dir, name)
        image = io.imread(img_name)
        # depth_img = io.imread(depth_img_name)
        # image = np.concatenate([image, depth_img]) # 4 * 1280* 720
        sample = {'image': image / 255 - 0.474, 'onion': np.array(label_dict[o_loc]),
                  'eef': np.array(label_dict[e_loc]), 'name': name}

        if self.transform:
            sample = self.transform(sample)

        return sample


class StateDatasetNew(Dataset):
    def __init__(self, csv_file, rgb_dir, transform=None, image_pool=None):
        self.labels = pd.read_csv(csv_file, header=None)
        self.rgb_dir = rgb_dir
        self.transform = transform
        self.image_pool = image_pool

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        name, e_loc, grasp = self.labels.iloc[idx]
        img_name = os.path.join(self.rgb_dir, name)
        image = io.imread(img_name)
        # depth_img = io.imread(depth_img_name)
        # image = np.concatenate([image, depth_img]) # 4 * 1280* 720
        sample = {'image': image / 255 - 0.474, 'eef': np.array(label_dict_new[e_loc]), 'grasp': np.array(grasp_dict[grasp]), 'name': name}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size=(480, 640)):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'onion': sample['onion'], 'eef': sample['eef'], 'name': sample['name']}


class RescaleAction(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size=(480, 640)):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[1:3]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = np.array([transform.resize(i, (new_h, new_w)) for i in image])

        return {'image': img, 'action': sample['action'], 'name': sample['name']}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, grasp, eef, name = sample['image'], sample['grasp'], sample['eef'], sample['name']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'grasp': torch.from_numpy(grasp),
                'eef': torch.from_numpy(eef),
                'name': name}


class ToTensorAction(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, action, name = sample['image'], sample['action'], sample['name']

        # swap color axis because
        # numpy image: L x H x W x C
        # torch image: L x C x H x W
        image = image.transpose((0, 3, 1, 2))
        return {'image': torch.from_numpy(image),
                'action': torch.from_numpy(action),
                'name': name}


if __name__ == "__main__":
    # state_model = StateDetect().to(device)
    #
    # # img = torch.rand((8, 3, 4, 480, 640))  # (batch, time sequence, channel(including depth), height, width)
    # # states = state_model(img[:, 0])
    # # action = action_model(img)
    #
    # transformed_dataset = StateDataset(csv_file='data/RealSense/state/label.csv',
    #                                    root_dir='data/RealSense/state/rgb',
    #                                    transform=transforms.Compose([
    #                                        Rescale(),
    #                                        ToTensor()]))
    #
    # dataloader = DataLoader(transformed_dataset, batch_size=32,
    #                         shuffle=True, num_workers=8)

    action_model = ActionDetect().to(device)

    transformed_dataset = ActionDataset(
        transform=transforms.Compose([
            RescaleAction(),
            ToTensorAction()
        ]))

    dataloader = DataLoader(transformed_dataset, batch_size=32,
                            shuffle=True, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(action_model.parameters())

    print('Start Training...')
    for epoch in range(10):  # loop over the dataset multiple times
        # train_loss_onion = []
        # train_loss_eef = []
        # train_pred_onion = []
        # train_label_onion = []
        # train_pred_eef = []
        # train_label_eef = []

        action_model.train()
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['image'].float().to(device)
            action = data['action'].to(device)
            # name = data['name']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = action_model(inputs)
            loss = criterion(outputs, action)
            loss.backward()
            optimizer.step()

            print(epoch, i, loss.item())

            # print statistics
            # train_loss_onion.append(loss_onion.item())
            # train_loss_eef.append(loss_eef.item())
            # train_pred_onion.extend(torch.argmax(outputs_onion, axis=1).cpu().numpy().tolist())
            # train_pred_eef.extend(torch.argmax(outputs_eef, axis=1).cpu().numpy().tolist())
            # train_label_onion.extend(onion.cpu().numpy().tolist())
            # train_label_eef.extend(eef.cpu().numpy().tolist())


