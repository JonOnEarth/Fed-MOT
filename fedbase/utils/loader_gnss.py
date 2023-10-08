import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os

# from utils.data_utils import read_client_data


# for jammer dataset, from non-FL github
def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('', dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True):
    # if dataset[:2] == "ag" or dataset[:2] == "SS":
    #     return read_client_data_text(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data
    
# load the minist dataset
def load_train_data(dataset,id):
    train_data = read_client_data(dataset, id, is_train=True)
    # return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)
    return train_data

def load_test_data(dataset,id):
    test_data = read_client_data(dataset, id, is_train=False)
    # return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
    return test_data

def load_dataloader(dataset_name, num_clients):
    train_loaders = []
    test_loaders = []
    for i in range(num_clients):
        train_loaders.append(load_train_data(dataset_name,i))
        test_loaders.append(load_test_data(dataset_name,i))
    # combine the test_loader of each client
    # # fix the bug of "local variable referenced before assignment"
    # test_x = None
    # test_label = None
    # for i in range(len(test_loaders)):
    #     # get the test_loader shape
    #     for x, y in test_loaders[i]:
    #         if i == 0:
    #             test_x = torch.empty(0, x.shape[1], x.shape[2], x.shape[3])
    #             test_label = torch.empty(0)
    #         test_x = torch.cat((test_x, x), 0) if test_x is not None else torch.empty(0, x.shape[1], x.shape[2], x.shape[3])
    #         test_label = torch.cat((test_label, y), 0) if test_label is not None else torch.empty(0)
    # test_label = test_label.long()
    # test_loader_global = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_label), args.test_batch_size, shuffle=False)
    split_para = None
    return train_loaders, test_loaders, split_para

def load_all(dataset_name, num_clients):
    train_data = []
    test_data = []
    for i in range(num_clients):
        train_data.extend(load_train_data(dataset_name,i))
        test_data.extend(load_test_data(dataset_name,i))
    return train_data, test_data
