import torchvision
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split, Subset, ChainDataset, ConcatDataset
import torch
from itertools import accumulate as _accumulate
import matplotlib.pyplot as plt
import matplotlib as mpl
from fedbase.utils.tools import get_targets
from fedbase.utils import femnist
import os
# import pickle
import json
import datetime as d
import math
import pandas as pd
from pathlib import Path
from collections import Counter
import torchvision.transforms.functional as TF
# import medmnist
from skimage.util import random_noise
import random
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

class data_process:
    def __init__(self, dataset_name):
        dir ='./data/'
        self.dataset_name = dataset_name
        if dataset_name == 'mnist':
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
            self.train_dataset = datasets.MNIST(
                dir+dataset_name, train=True, download=True, transform=apply_transform)
            self.test_dataset = datasets.MNIST(
                dir+dataset_name, train=False, download=True, transform=apply_transform)
        elif dataset_name == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.train_dataset = datasets.CIFAR10(
                dir+dataset_name, train=True, download=True, transform=transform)
            self.test_dataset = datasets.CIFAR10(
                dir+dataset_name, train=False, download=True, transform=transform)
        elif dataset_name == 'femnist':
            apply_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            self.train_dataset = femnist.FEMNIST(dir+dataset_name, train=True, download=False,
                                            transform=apply_transform)
            self.test_dataset = femnist.FEMNIST(dir+dataset_name, train=False, download=False,
                                           transform=apply_transform)
        elif dataset_name == 'fashion_mnist':
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])
            self.train_dataset = datasets.FashionMNIST(
                dir+dataset_name, train=True, download=True, transform=apply_transform)
            self.test_dataset = datasets.FashionMNIST(
                dir+dataset_name, train=False, download=True, transform=apply_transform)
        elif 'medmnist' in dataset_name:
            data_flag = dataset_name[9:]
            DataClass = getattr(medmnist, medmnist.INFO[data_flag]['python_class'])
            # preprocessing
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])
            ])
            # load the data
            dir = dir + dataset_name +'/'
            if not os.path.exists(dir):
                os.mkdir(dir)
            self.train_dataset = DataClass(split='train', transform=data_transform, download=True, root = dir)
            # self.train_dataset.labels = torch.tensor(self.train_dataset.labels, dtype = torch.long)
            self.val_dataset = DataClass(split='val', transform=data_transform, download=True, root = dir)
            self.test_dataset = DataClass(split='test', transform=data_transform, download=True, root = dir)

            self.test_dataset = ConcatDataset([self.val_dataset, self.test_dataset])
            # print(len(self.val_dataset), len(self.test_dataset))
            # print(self.train_dataset)
        elif dataset_name == 'jammer':
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            # self.train_dataset = datasets.ImageFolder(
            #     dir+dataset_name+'/jammer_train_dataset', transform=transform)
            # self.test_dataset = datasets.ImageFolder(
            #     dir+dataset_name+'/jammer_test_dataset', transform=transform)
            
            # choose 1000 images from each class of folder as the train dataset
            # Specify the paths to the folders containing the images
            # construct the full dataset

            train_dataset = ImageFolder(dir+dataset_name+'/Image_training_database', transform=transform)
            test_dataset = ImageFolder(dir+dataset_name+'/Image_training_database', transform=transform)
            # select the 1000 indices of each folder
            def subset_indices(dataset,size=100):
                idxs = []
                for i in range(6):
                    idx = [j for j in range(len(dataset)) if dataset.imgs[j][1] == i]
                    # random choose 1000 indices
                    idx = random.sample(idx, size)
                    idxs.extend(idx)
                    # print(idx[1:10])
                return idxs
            # build the appropriate subset
            self.train_dataset = Subset(train_dataset, subset_indices(train_dataset))
            self.test_dataset = Subset(test_dataset, subset_indices(test_dataset))
            print('train dataset size: ', len(self.train_dataset))
            print('test dataset size: ', len(self.test_dataset))
            # # choose 1000 images from each class
            # chunk_size = 1000
            # subsets = {target: Subset(self.train_dataset, [i for i, (x, y) in enumerate(self.train_dataset) if y == target]) for _, target in self.train_dataset.class_to_idx.items()}
            # self.train_dataset = ConcatDataset([Subset(subset, np.random.choice(len(subset), chunk_size, replace=False)) for subset in subsets.values()])
            # print(len(self.train_dataset))
            # chunk_size_test = 200
            # subsets_test = {target: Subset(self.test_dataset, [i for i, (x, y) in enumerate(self.test_dataset) if y == target]) for _, target in self.test_dataset.class_to_idx.items()}
            # self.test_dataset = ConcatDataset([Subset(subset, np.random.choice(len(subset), chunk_size_test, replace=False)) for subset in subsets_test.values()])
            # print(len(self.test_dataset))

            # # shuffle the dataset
            # total_size = len(self.train_dataset)
            # indices = list(range(total_size))
            # np.random.shuffle(indices)
            # self.train_dataset = Subset(self.train_dataset, indices[:10000])
            # total_size_test = len(self.test_dataset)
            # indices_test = list(range(total_size_test))
            # np.random.shuffle(indices_test)
            # self.test_dataset = Subset(self.test_dataset, indices_test[:2000])
            # print(len(self.train_dataset))

        sample = next(iter(self.train_dataset))
        image, label = sample
        print(image.shape)

        # show image
        # batch_size = 4
        # trainloader = DataLoader(self.train_dataset, batch_size=batch_size,
        #                                   shuffle=True, num_workers=2)
        # classes = ('plane', 'car', 'bird', 'cat',
        #    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # def imshow(img):
        #     img = img / 2 + 0.5     # unnormalize
        #     npimg = img.numpy()
        #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
        #     plt.show()

        # # get some random training images
        # dataiter = iter(trainloader)
        # images, labels = dataiter.next()

        # # show images
        # imshow(torchvision.utils.make_grid(images))
        # # print labels
        # print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    def split_dataset(self, num_nodes, alpha, method='dirichlet',noise=None, train_dataset = None, test_dataset = None, plot_show = False):
        train_dataset = self.train_dataset if train_dataset is None else train_dataset
        test_dataset = self.test_dataset if test_dataset is None else test_dataset
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
                self.plot_split(labels, train_splited)
                # add noise
                if noise == 'rotation':
                    for i in range(len(train_splited)):
                        # for j in range(len(train_splited[i])):
                        for j,(img, label) in enumerate(train_splited[i]):
                            img = TF.rotate(img, 45)
                            train_splited[i][j] = (img, label)
                    for i in range(len(test_splited)):
                        # for j in range(len(test_splited[i])):
                        for j,(img, label) in enumerate(test_splited[i]):
                            img = TF.rotate(img, 45)
                            # replace the new img to the dataset
                            test_splited[i][j] = (img, label)
                elif noise == 'Gaussian':
                    for i in range(len(train_splited)):
                        if i==0:
                            continue
                        var = 0.1  * i
                        for j,(img, label) in enumerate(train_splited[i]):
                            img = torch.tensor(random_noise(img, mode='gaussian', mean=0., var=var, clip=True))
                            train_splited[i][j] = (img, label)
                    for i in range(len(test_splited)):
                        if i==0:
                            continue
                        var = 0.1  * i
                        for j,(img, label) in enumerate(test_splited[i]):
                            img = torch.tensor(random_noise(img, mode='gaussian', mean=0., var=var, clip=True))
                            test_splited[i][j] = (img, label)
                return train_splited, test_splited
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
                if plot_show:
                    self.plot_split(labels, train_splited)
                # print(min([len(i) for i in train_splited]))
                # print(min([len(i) for i in test_splited]))
                if noise == 'rotation':
                    for i in range(len(train_splited)):
                        # for j in range(len(train_splited[i])):
                        # for j,(img, label) in enumerate(train_splited[i]):
                        #     img = TF.rotate(img, 45)
                        #     train_splited[i][j] = img, label
                        for j in range(len(train_splited[i])):
                            img, label = train_splited[i][j]
                            img = TF.rotate(img, 45)
                            train_splited[i][j] = img, label
                    for i in range(len(test_splited)):
                        # for j in range(len(test_splited[i])):
                        for j,(img, label) in enumerate(test_splited[i]):
                            img = TF.rotate(img, 45)
                            # replace the new img to the dataset
                            test_splited[i][j] = (img, label)
                elif noise == 'Gaussian':
                    for i in range(len(train_splited)):
                        if i == 0:
                            continue
                        var = i*0.1
                        for j,(img, label) in enumerate(train_splited[i]):
                            img = torch.tensor(random_noise(img, mode='gaussian', mean=0., var=var, clip=True))
                            train_splited[i][j] = (img, label)
                    for i in range(len(test_splited)):
                        if i == 0:
                            continue
                        var = i*0.1
                        for j,(img, label) in enumerate(test_splited[i]):
                            img = torch.tensor(random_noise(img, mode='gaussian', mean=0., var=var, clip=True))
                            test_splited[i][j] = (img, label)
                return train_splited, test_splited, self.dataset_name +'_'+ str(num_nodes)+'_'+ str(alpha)+'_'+ str(method)+'_'+ str(noise)
        
    def split_dataset_groupwise(self, num0, alpha0, method0, num1, alpha1, method1, noise=None, train_dataset = None, test_dataset = None, plot_show = False):
        train_dataset = self.train_dataset if train_dataset is None else train_dataset
        test_dataset = self.test_dataset if test_dataset is None else test_dataset
        train_targets = get_targets(train_dataset)
        train_splited = []
        test_splited = []
        # to control min length of group dataset
        train_splited_0, test_splited_0, _ = self.split_dataset(num0, alpha0, method0, noise=noise)
        while (min([len(test_splited_0[i]) for i in range(len(test_splited_0))]) <= len(test_dataset)/num0 * 0.3):
            # print('do it again')
            train_splited_0, test_splited_0, _ = self.split_dataset(num0, alpha0, method0, noise=noise)
        for i in range(num0):
            train_tmp, test_tmp, _ = self.split_dataset(num1, alpha1, method1, train_dataset=train_splited_0[i], test_dataset=test_splited_0[i])
            train_splited += train_tmp
            test_splited += test_tmp
        #plot
        labels = torch.unique(train_targets)
        if plot_show:
            self.plot_split(labels, train_splited)
        return train_splited, test_splited, self.dataset_name +'_'+ str(num0)+'_'+ str(alpha0)+'_'+ str(method0)+'_'+ str(num1)+'_'+ str(alpha1)+'_'+ str(method1)

    def plot_split(self, labels, train_splited):
        # return None
        train_size = []
        for x in train_splited:
            tmp = []
            train_classes = [int(label) for _, label in x]
            for i in range(len(labels)):
                tmp.append(train_classes.count(labels[i]))
            # train_targets = get_targets(x)
            # labels, train_label_size = torch.unique(train_targets, return_counts=True)
            train_size.append(tmp)
        train_size = torch.tensor(train_size).T
        # plot
        for i in range(len(labels)):
            plt.barh([j for j in range(1,len(train_splited)+1)], train_size[i, :], left=torch.sum(
                train_size[:i], 0), label=str(int(labels[i])))
        # plt.title("Data distribution of dataset")
        plt.legend()
        plt.title('Client-wise Non-IID Setting', fontsize = 20)
        # plt.title('Cluster-wise Non-IID Setting', fontsize = 20)
        plt.xlabel('Dataset size', fontsize=16)
        plt.ylabel('Client ID', fontsize=16)
        # plt.ylim((0,len(train_splited)-1))
        plt.show()

    # def add_rotation_noise(train_data,label, alpha):
    #     train_data_rotated = []
    #     for i in range(len(train_data)):
    #         train_data_rotated.append(train_data[i].rotate(alpha))
    #     plt.imshow(train_data_rotated[0][0].squeeze().numpy(), cmap='gray')
    #     plt.show()

        # return train_data_rotated


def log(file_name, nodes, server, H=None,assign_method=None, bayes=None,path='log/'):
    local_file = './'+path + file_name + "_" + d.datetime.now().strftime("%m%d_%H%M%S")+'_'+str(np.random.choice(10**3)) + ".json"
    log = {}
    log['node'] = {}
    for i in range(len(nodes)):
        log['node'][str(i)] = list(nodes[i].test_metrics)
        # log['node_confusion_matrix'] = list(nodes[i].con_mats)
    try:
        log['server'] = list(server.test_metrics)
        log['best_assignment'] = list(server.test_metrics_best)
        log['clustering'] = str(server.clustering)
        log['assign_method'] = str(assign_method)
        log['bayes'] = str(bayes)
        log['H'] = str(H)
        # log['confusion_matrix'] = list(server.con_mats)
        # con_mats is a numpy array, can not be saved in json,how to save it?
        # transfer con_mats to a format can be saved in json
        log['confusion_matrix'] = []
        for i in range(len(server.con_mats)):
            log['confusion_matrix'].append(server.con_mats[i].tolist())

    except:
        print('No server')
    # pd.to_pickle(log, local_file)
    # print(log)
    Path(local_file).parent.mkdir(parents=True, exist_ok=True)
    with  open(local_file, 'w') as handle:
        json.dump(log, handle, indent=4)
    # read
    # if os.path.exists(local_file):
    #     with open(local_file, 'r') as f:
    #         log = json.load(f)
    #         # print(log)

# dt = data_process('mnist')
# # dt.split_dataset(50, 2, method='class')
# dt.split_dataset(10, 0.1)


# def data_split(dataset, num_nodes, type='iid'):
#     if type == 'iid':
#         lens_list = [int(len(dataset)/num_nodes) for i in range(num_nodes-1)]
#         lens_list.append(len(dataset)-sum(lens_list))
#         return random_split(dataset, lens_list)
# def data_split_noniid(dataset, num_nodes, num_category):


# def data_split(dataset, num, type='iid'):
#     num_items = int(len(dataset)/num)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users


# class DatasetSplit(Dataset):
#     def __init__(self, dataset, idxs):
#         self.dataset = dataset
#         self.idxs = list(idxs)

#     def __len__(self):
#         return len(self.idxs)

#     def __getitem__(self, item):
#         image, label = self.dataset[self.idxs[item]]
#         return image, label



