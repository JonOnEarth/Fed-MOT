import numpy as np
import os
import random
import torchvision.transforms as transforms
import torch.utils.data as data
# from utils.dataset_utils import split_data, save_file
from os import path
from scipy.io import loadmat
from PIL import Image
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset, ChainDataset, ConcatDataset
from fedbase.utils.tools import get_targets
from itertools import accumulate as _accumulate
from sklearn.model_selection import train_test_split


class CustomImageDataset(data.Dataset):
    def __init__(self,img_data, img_labels, transform=None, target_transform=None):
        super(CustomImageDataset, self).__init__()
        self.data = img_data
        self.labels = img_labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label
    
def group_split(train_datasets, test_datasets, num_clients_per_group, method='iid', alpha=None):
    # train_targets = get_targets(train_datasets[0])
    train_splited = []
    test_splited = []
    for i in range(len(train_datasets)):
        # transform = transforms.Compose([
        #         # you can add other transformations in this list
        #         transforms.ToTensor()
        #     ])
        # train_dataset = Digit5Dataset(data=train_datasets[i]['data'], labels=train_datasets[i]['labels'], transform=transform)
        # test_dataset = Digit5Dataset(data=test_datasets[i]['data'], labels=test_datasets[i]['labels'], transform=transform)
        train_dataset = CustomImageDataset(train_datasets[i]['data'], train_datasets[i]['labels'])
        test_dataset = CustomImageDataset(test_datasets[i]['data'], test_datasets[i]['labels'])
        print('Done loader!')
        train_tmp, test_tmp = split_dataset(num_clients_per_group, alpha, method, train_dataset=train_dataset, test_dataset=test_dataset)
        train_splited += train_tmp
        test_splited += test_tmp
    #plot
    # labels = torch.unique(train_targets)
    
    return train_splited, test_splited

def split_dataset(num_nodes, alpha, method='dirichlet', train_dataset = None, test_dataset = None, plot_show = False):
    # train_dataset = self.train_dataset if train_dataset is None else train_dataset
    # test_dataset = self.test_dataset if test_dataset is None else test_dataset
    train_targets, test_targets = get_targets(train_dataset), get_targets(test_dataset)
    if num_nodes == 1:
        return train_dataset, test_dataset
    else:
        if method == 'iid':
            train_lens_list = [int(len(train_dataset)/num_nodes) for i in range(num_nodes)]
            test_lens_list = [int(len(test_dataset)/num_nodes) for i in range(num_nodes)]
            train_splited, test_splited = random_split(Subset(train_dataset, torch.arange(sum(train_lens_list))), train_lens_list), random_split(Subset(test_dataset, torch.arange(sum(test_lens_list))), test_lens_list)
            # plot
            labels = torch.unique(train_targets)
            # self.plot_split(labels, train_splited)
            # add noise
        else:
            labels, train_label_size = torch.unique(train_targets, return_counts=True)
            _, test_label_size = torch.unique(test_targets, return_counts=True)
            # print(train_label_size, test_label_size)
            l_train = train_label_size.reshape(
                len(labels), 1).repeat(1, num_nodes)
            l_test = test_label_size.reshape(
                len(labels), 1).repeat(1, num_nodes)
            
            train_splited = []
            test_splited = []
            while len(test_splited) <= num_nodes//2:
                # print(l_test)
                if method == 'dirichlet':
                    # print(len(test_dataset), min(test_label_size))
                    # dec_round = round(math.log(len(test_dataset)/len(labels),10))
                    dec_round = 2
                    # p = torch.tensor(np.round(np.random.dirichlet(np.repeat(alpha, num_nodes), len(labels)), round(math.log(len(test_dataset)/len(labels),10))))
                    p = torch.tensor(np.floor(np.random.dirichlet(np.repeat(alpha, num_nodes), len(labels))*10**dec_round)/10**dec_round)
                    # print(torch.sum(p,axis=1))
                    # print(p)
                elif method == 'class':
                    p = np.zeros((len(labels), 1))
                    J = np.random.choice(len(labels), alpha, replace=False)
                    p[J] = 1
                    for k in range(1, num_nodes):
                        x = np.zeros((len(labels), 1))
                        J = np.random.choice(len(labels), alpha, replace=False)
                        x[J] = 1
                        p = np.concatenate((p, x), axis=1)
                    p = p / np.repeat((p.sum(axis=1)+10**-10).reshape(len(labels), 1), num_nodes, axis=1)
                # print(p.sum(axis=1),p)
                train_size = torch.round(l_train*p).int()
                test_size = torch.round(l_test*p).int()
                # print(train_size, test_size)
                train_label_index = []
                test_label_index = []
                for j in range(len(labels)):
                    train_label_index.append([(train_targets== labels[j]).nonzero(as_tuple=True)[
                                            0][offset-length:offset] for offset, length in zip(_accumulate(train_size[j, :]), train_size[j, :])])
                    test_label_index.append([(test_targets== labels[j]).nonzero(as_tuple=True)[
                                            0][offset-length:offset] for offset, length in zip(_accumulate(test_size[j, :]), test_size[j, :])])
                # how to deal with 0?
                for i in range(num_nodes):
                    if len(ConcatDataset([Subset(test_dataset, test_label_index[j][i]) for j in range(len(labels))]))>5: # 0-10, to control the minimun length
                        train_splited.append(ConcatDataset(
                            [Subset(train_dataset, train_label_index[j][i]) for j in range(len(labels))]))
                        test_splited.append(ConcatDataset(
                            [Subset(test_dataset, test_label_index[j][i]) for j in range(len(labels))]))
            while len(test_splited)<num_nodes:
                # print(self.dataset_name,len(test_splited),num_nodes-len(test_splited))
                random_index = np.random.choice(range(len(test_splited)), num_nodes-len(test_splited), replace=True)
                train_splited = train_splited + [train_splited[i] for i in range(len(train_splited)) if i in random_index]           
                test_splited = test_splited + [test_splited[i] for i in range(len(test_splited)) if i in random_index]  
            # if plot_show:
            #     self.plot_split(labels, train_splited)
            # print(min([len(i) for i in train_splited]))
            # print(min([len(i) for i in test_splited]))
            
        return train_splited, test_splited
        
def split_data(X, y,train_size=0.75):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_size, shuffle=True)

        train_data.append({'data': X_train, 'labels': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'data': X_test, 'labels': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    return train_data, test_data

