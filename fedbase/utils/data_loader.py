import torchvision
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split, Subset, ChainDataset, ConcatDataset
import torch
from torch._utils import _accumulate
import matplotlib.pyplot as plt
import matplotlib as mpl
from fedbase.utils import femnist
import os
import pickle
import datetime as d


class data_process:
    def __init__(self, dir, dataset_name):
        if dataset_name == 'mnist':
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
            self.train_dataset = datasets.MNIST(
                dir, train=True, download=True, transform=apply_transform)
            self.test_dataset = datasets.MNIST(
                dir, train=False, download=True, transform=apply_transform)
        elif dataset_name == 'cifar10':
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.train_dataset = datasets.CIFAR10(root='./data/'+dataset_name, train=True,download=True, transform=transform)
            self.test_dataset = datasets.CIFAR10(root='./data/'+dataset_name, train=False,download=True, transform=transform)
        elif dataset_name == 'femnist':
            apply_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])    
            train_dataset = femnist.FEMNIST( './data/'+dataset_name, train=True, download=False,
                                        transform=apply_transform)
            test_dataset = femnist.FEMNIST( './data/'+dataset_name, train=False, download=False,
                                       transform=apply_transform)
        elif dataset_name == 'Fashion_Mnist':
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])
            self.train_dataset = datasets.FashionMNIST(
                './data/'+dataset_name, train=True, download=True, transform=apply_transform)
            self.test_dataset = datasets.FashionMNIST(
                './data/'+dataset_name, train=False, download=True, transform=apply_transform)


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
        

    def split_dataset(self, num_nodes, alpha, method = 'dirichlet'):
        if num_nodes ==1:
            return self.train_dataset,self.test_dataset
        else:
            if method == 'iid':
                lens_list = [int(len(dataset)/num_nodes) for i in range(num_nodes-1)]
                lens_list.append(len(dataset)-sum(lens_list))
                return random_split(dataset, lens_list)
            else:
                self.train_dataset.targets = torch.tensor(self.train_dataset.targets)
                self.test_dataset.targets = torch.tensor(self.test_dataset.targets)
                labels = torch.unique(self.train_dataset.targets)
                train_label_size = self.train_dataset.targets.bincount()
                test_label_size = self.test_dataset.targets.bincount()
                l_train = train_label_size.reshape(len(labels), 1).repeat(1, num_nodes)
                l_test = test_label_size.reshape(len(labels), 1).repeat(1, num_nodes)
                # print(l_test)
                if method == 'dirichlet':
                    p = torch.tensor(np.random.dirichlet(np.repeat(alpha, num_nodes), len(labels)))
                elif method == 'class':
                    p = np.zeros((len(labels),1))
                    J=np.random.choice(len(labels),alpha, replace=False)
                    p[J]=1
                    for k in range(1,num_nodes):
                        x = np.zeros((len(labels),1))
                        J=np.random.choice(len(labels),alpha, replace=False)
                        x[J]=1
                        p=np.concatenate((p,x),axis=1)
                    p= p/np.repeat((p.sum(axis=1)+10**-10).reshape(len(labels),1),num_nodes,axis=1)
                # print(p.sum(axis=1),p)
                # p = torch.tensor(np.random.dirichlet(np.repeat(alpha,len(labels)), num_nodes)).reshape(10,100)
                # for i in range(len(p)):
                #     if l_test*p[i]<10:
                #         p[i]+=0.01
                #         p[p==p.max()]-=0.01
                train_size = l_train*p
                train_size = train_size.int()
                test_size = l_test*p
                test_size = test_size.int()
                # print(torch.sum(train_size,0),torch.sum(test_size,0))
                # print(len((self.train_dataset.targets==0).nonzero(as_tuple=True)[0]))
                train_splited =[]
                test_splited =[]
                train_label_index = []
                test_label_index = []
                for j in labels:
                    train_label_index.append([(self.train_dataset.targets==j).nonzero(as_tuple=True)[0][offset-length:offset] for offset,length in zip(_accumulate(train_size[j,:]), train_size[j,:])])
                    test_label_index.append([(self.test_dataset.targets==j).nonzero(as_tuple=True)[0][offset-length:offset] for offset,length in zip(_accumulate(test_size[j,:]), test_size[j,:])])
                for i in range(num_nodes):
                    train_splited.append(ConcatDataset([Subset(self.train_dataset,train_label_index[j][i]) for j in labels]))
                    test_splited.append(ConcatDataset([Subset(self.test_dataset,test_label_index[j][i]) for j in labels]))
                # print([len(train_splited[i]) for i in range(num_nodes)],[len(train_splited[i]) for i in range(num_nodes)])
                # plot
                for i in labels:
                    plt.barh(range(num_nodes), train_size[i,:], left=torch.sum(train_size[:i],0),label=str(int(i)))
                # plt.title("Data distribution of dataset")
                plt.legend()
                mpl.rcParams['figure.dpi'] = 300
                plt.show()
                return train_splited, test_splited


# dt = data_process('mnist')
# dt.split_dataset(50, 2, method='class')
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


def log(file_name, nodes, server):
    local_file = './log/' + file_name + '_' + str(d.datetime.now()) + '.pkl'
    log = {}
    log['node'] = {}
    for i in range(len(nodes)):
        log['node'][str(i)]=nodes[i].accuracy
    log['server']=server.accuracy
    log['clustering']=server.clustering
    pickle.dump(log, open(local_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    # read
    if os.path.exists(local_file):
        log = pickle.load(open(local_file, 'rb'))
        print(log)