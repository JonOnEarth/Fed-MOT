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
from torch._utils import _accumulate

# https://github.com/FengHZ/KD3A/blob/master/datasets/DigitFive.py
def load_mnist(base_path):
    print("load mnist")
    mnist_data = loadmat(path.join(base_path, "mnist_data.mat"))
    mnist_train = np.reshape(mnist_data['train_32'], (55000, 32, 32, 1))
    mnist_test = np.reshape(mnist_data['test_32'], (10000, 32, 32, 1))
    # turn to the 3 channel image with C*H*W
    mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
    mnist_train = mnist_train.transpose(0, 3, 1, 2).astype(np.float32)
    mnist_test = mnist_test.transpose(0, 3, 1, 2).astype(np.float32)
    # get labels
    mnist_labels_train = mnist_data['label_train']
    mnist_labels_test = mnist_data['label_test']
    # random sample 25000 from train dataset and random sample 9000 from test dataset
    train_label = np.argmax(mnist_labels_train, axis=1)
    inds = np.random.permutation(mnist_train.shape[0])
    mnist_train = mnist_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnist_labels_test, axis=1)

    mnist_train = mnist_train[:25000]
    train_label = train_label[:25000]
    mnist_test = mnist_test[:9000]
    test_label = test_label[:9000]
    return mnist_train, train_label, mnist_test, test_label


def load_mnist_m(base_path):
    print("load mnist_m")
    mnistm_data = loadmat(path.join(base_path, "mnistm_with_label.mat"))
    mnistm_train = mnistm_data['train']
    mnistm_test = mnistm_data['test']
    mnistm_train = mnistm_train.transpose(0, 3, 1, 2).astype(np.float32)
    mnistm_test = mnistm_test.transpose(0, 3, 1, 2).astype(np.float32)
    # get labels
    mnistm_labels_train = mnistm_data['label_train']
    mnistm_labels_test = mnistm_data['label_test']
    # random sample 25000 from train dataset and random sample 9000 from test dataset
    train_label = np.argmax(mnistm_labels_train, axis=1)
    inds = np.random.permutation(mnistm_train.shape[0])
    mnistm_train = mnistm_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnistm_labels_test, axis=1)
    mnistm_train = mnistm_train[:25000]
    train_label = train_label[:25000]
    mnistm_test = mnistm_test[:9000]
    test_label = test_label[:9000]
    return mnistm_train, train_label, mnistm_test, test_label


def load_svhn(base_path):
    print("load svhn")
    svhn_train_data = loadmat(path.join(base_path, "svhn_train_32x32.mat"))
    svhn_test_data = loadmat(path.join(base_path, "svhn_test_32x32.mat"))
    svhn_train = svhn_train_data['X']
    svhn_train = svhn_train.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_test = svhn_test_data['X']
    svhn_test = svhn_test.transpose(3, 2, 0, 1).astype(np.float32)
    train_label = svhn_train_data["y"].reshape(-1)
    test_label = svhn_test_data["y"].reshape(-1)
    inds = np.random.permutation(svhn_train.shape[0])
    svhn_train = svhn_train[inds]
    train_label = train_label[inds]
    svhn_train = svhn_train[:25000]
    train_label = train_label[:25000]
    svhn_test = svhn_test[:9000]
    test_label = test_label[:9000]
    train_label[train_label == 10] = 0
    test_label[test_label == 10] = 0
    return svhn_train, train_label, svhn_test, test_label


def load_syn(base_path):
    print("load syn")
    syn_train_data = loadmat(path.join(base_path, "synth_train_32x32.mat"))
    syn_test_data = loadmat(path.join(base_path, "synth_test_32x32.mat"))
    syn_train = syn_train_data["X"]
    syn_test = syn_test_data["X"]
    syn_train = syn_train.transpose(3, 2, 0, 1).astype(np.float32)
    syn_test = syn_test.transpose(3, 2, 0, 1).astype(np.float32)
    train_label = syn_train_data["y"].reshape(-1)
    test_label = syn_test_data["y"].reshape(-1)
    syn_train = syn_train[:25000]
    syn_test = syn_test[:9000]
    train_label = train_label[:25000]
    test_label = test_label[:9000]
    train_label[train_label == 10] = 0
    test_label[test_label == 10] = 0
    return syn_train, train_label, syn_test, test_label


def load_usps(base_path):
    print("load usps")
    usps_dataset = loadmat(path.join(base_path, "usps_28x28.mat"))
    usps_dataset = usps_dataset["dataset"]
    usps_train = usps_dataset[0][0]
    train_label = usps_dataset[0][1]
    train_label = train_label.reshape(-1)
    train_label[train_label == 10] = 0
    usps_test = usps_dataset[1][0]
    test_label = usps_dataset[1][1]
    test_label = test_label.reshape(-1)
    test_label[test_label == 10] = 0
    usps_train = usps_train * 255
    usps_test = usps_test * 255
    usps_train = np.concatenate([usps_train, usps_train, usps_train], 1)
    usps_train = np.tile(usps_train, (4, 1, 1, 1))
    train_label = np.tile(train_label,4)
    usps_train = usps_train[:25000]
    train_label = train_label[:25000]
    usps_test = np.concatenate([usps_test, usps_test, usps_test], 1)
    return usps_train, train_label, usps_test, test_label

class Digit5Dataset(data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        super(Digit5Dataset, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        if img.shape[0] != 1:
            # transpose to Image type,so that the transform function can be used
            img = Image.fromarray(np.uint8(np.asarray(img.transpose((1, 2, 0)))))

        elif img.shape[0] == 1:
            im = np.uint8(np.asarray(img))
            # turn the raw image into 3 channels
            im = np.vstack([im, im, im]).transpose((1, 2, 0))
            img = Image.fromarray(im)

        # do transform with PIL
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return self.data.shape[0]

def digit5_dataset_read(base_path, domain):
    if domain == "mnist":
        train_image, train_label, test_image, test_label = load_mnist(base_path)
    elif domain == "mnistm":
        train_image, train_label, test_image, test_label = load_mnist_m(base_path)
    elif domain == "svhn":
        train_image, train_label, test_image, test_label = load_svhn(base_path)
    elif domain == "syn":
        train_image, train_label, test_image, test_label = load_syn(base_path)
    elif domain == "usps":
        train_image, train_label, test_image, test_label = load_usps(base_path)
    else:
        raise NotImplementedError("Domain {} Not Implemented".format(domain))
    # define the transform function
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # raise train and test data loader
    train_dataset = Digit5Dataset(data=train_image, labels=train_label, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_dataset = Digit5Dataset(data=test_image, labels=test_label, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
    return train_loader, test_loader


random.seed(1)
np.random.seed(1)
data_path = "Digit5/"
dir_path = "Digit5/"

# Allocate data to usersz``
def generate_Digit5(dir_path, domains = ['mnistm', 'mnist', 'syn', 'usps', 'svhn'], client_group=1, method='iid',alpha=0.5):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    root = data_path+"rawdata"
    
    # Get Digit5 data
    if not os.path.exists(root):
        os.makedirs(root)
        os.system(f'wget https://drive.google.com/u/0/uc?id=1PT6K-_wmsUEUCxoYzDy0mxF-15tvb2Eu&export=download -P {root}')
        os.system(f'unzip {root}/Digit5.zip -d {root}')

    X, y = [], []
    # domains = ['mnistm', 'mnist', 'syn', 'usps', 'svhn']
    for d in domains:
        train_loader, test_loader = digit5_dataset_read(root, d)

        for _, tt in enumerate(train_loader):
            train_data, train_label = tt
        for _, tt in enumerate(test_loader):
            test_data, test_label = tt

        dataset_image = []
        dataset_label = []

        dataset_image.extend(train_data.cpu().detach().numpy())
        dataset_image.extend(test_data.cpu().detach().numpy())
        dataset_label.extend(train_label.cpu().detach().numpy())
        dataset_label.extend(test_label.cpu().detach().numpy())

        X.append(np.array(dataset_image))
        y.append(np.array(dataset_label))
    

    labelss = []
    for yy in y:
        labelss.append(len(set(yy)))
    num_clients = len(y)
    print(f'Number of labels: {labelss}')
    print(f'Number of clients: {num_clients}')

    statistic = [[] for _ in range(num_clients)]
    for client in range(num_clients):
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))


    train_data_groups, test_data_groups = split_data(X, y)
    # split train data and test data into different clients
    train_data, test_data = group_split(train_data_groups, test_data_groups, client_group, method=method, alpha=alpha)

    # save_file(config_path, train_path, test_path, train_data, test_data, num_clients, max(labelss), 
    #     statistic, None, None, None)
    
    return train_data, test_data

def group_split(train_datasets, test_datasets, num_clients_per_group, method='iid', alpha=None):
    # train_targets = get_targets(train_datasets[0])
    train_splited = []
    test_splited = []
    for i in range(len(train_datasets)):
        train_tmp, test_tmp, _ = split_dataset(num_clients_per_group, alpha, method, train_dataset=train_datasets[i], test_dataset=test_datasets[i])
        train_splited += train_tmp
        test_splited += test_tmp
    #plot
    # labels = torch.unique(train_targets)
    
    return train_splited, test_splited, 'digit5' +'_'+ str(num_clients_per_group)+'_'+ str(alpha)+'_'+ str(method)

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
            
            return train_splited, test_splited, 'digit5' +'_'+ str(num_nodes)+'_'+ str(alpha)+'_'+ str(method)
        
def split_data(X, y):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_size, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
if __name__ == "__main__":
    generate_Digit5(dir_path)