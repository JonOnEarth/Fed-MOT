import os
from fedbase.baselines import *
from fedbase.model.model import *
from fedbase.nodes.node import node
from fedbase.utils.tools import unpack_args
from fedbase.utils.data_loader import data_process
import torch
import torch.optim as optim
import torch.nn as nn
from functools import partial
import numpy as np
import multiprocessing as mp
import time
import torchvision.models as models

from joblib import Parallel, delayed
import argparse
from fedbase.utils.get_digit5 import generate_Digit5
from fedbase.utils.get_amazon_review import generate_AmazonReview
from fedbase.utils.get_domainnet import generate_DomainNet

os.chdir(os.path.dirname(os.path.abspath(__file__))) # set the current path as the working directory
global_rounds = 100
# num_nodes = 10
local_steps = 10
batch_size = 32
optimizer = partial(optim.SGD,lr=0.005, momentum=0.9)
# optimizer = partial(optim.SGD,lr=0.001)
# device = torch.device('cuda:2')
# device = torch.device('cuda')  # Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
K = 4
H = 2

def main(seeds, dataset_splited, model, model_name, K=None,n_assign=None,cost_method='weighted',warm_up=False,warm_up_rounds=2):
    np.random.seed(seeds)
    # dataset_splited, model = dataset_splited_model[0], dataset_splited_model[1]
    if model_name == 'FedAvg':
        fedavg.run(dataset_splited, batch_size, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, device = device)
    elif model_name == 'BayesFedAvg':
        fedavg_bayes.run(dataset_splited,batch_size, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, device = device)
    elif model_name == 'BayesPFedAvg':
        fedavg_bayes.run(dataset_splited,batch_size, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, device = device, personalized=True)
    elif model_name == 'Wecfl':
        GNN.run(dataset_splited,batch_size,K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, assign_method='wecfl',bayes=False, device = device)
    elif model_name == 'Fesem':
        fesem.run(dataset_splited,batch_size, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, reg_lam=0.001, device = device)
    elif model_name == 'GNN':
        GNN.run(dataset_splited,batch_size,K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, assign_method='ifca',bayes=True, device = device,warm_up=warm_up)
    elif model_name == 'JPDA':
        jpda.run(dataset_splited, batch_size, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, bayes=True, num_assign=n_assign,device=device, cost_method=cost_method,warm_up=warm_up)
    elif model_name == 'MHT':
        mht.run(dataset_splited, batch_size, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, bayes=True, num_assign=n_assign,hypothesis=n_assign, device=device, cost_method=cost_method,warm_up=warm_up)
    elif model_name == 'FedAMP':
        fedamp.run(dataset_splited, batch_size, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, device = device)
    elif model_name == 'FedSoft':
        fedsoft.run(dataset_splited, batch_size, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, selection_size=15,num_classes=10,reg_lam=0.1,device = device)
    elif model_name == 'FedEM':
        fedem.run(dataset_splited, batch_size, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, num_learners=K, device=device)
if __name__ == '__main__':
    dataset = 'cifar10' #'amazon' #'digit5'
    seeds = 1989 # 0,2020
    if dataset == 'mnist':
        model = CNNMnist
    elif dataset == 'femnist':
        model = CNNFemnist
    elif dataset == 'fashion_mnist':
        model = CNNFashion_Mnist
    elif dataset == 'cifar10':
        model = CNNCifar
    elif dataset == 'digit5':
        model = Digit5CNN # CNNDigit5
    elif dataset == 'amazon':
        model = AmazonMLP
    elif dataset == 'domainnet':
        model = AlexNet

    client_group = 10
    if dataset == 'digit5':
        # for Digit5
        domains=['mnistm', 'mnist', 'usps', 'svhn','syn'] #,
        K = len(domains)
        num_nodes = K*client_group #40
        dataset_splited_list = [generate_Digit5(domains=domains, client_group=client_group, method='iid', alpha=10)]
    elif dataset == 'amazon':
        K = 4
        num_nodes = K*client_group
        dataset_splited_list = [generate_AmazonReview(client_group=client_group, method='iid', alpha=10)]
    elif dataset == 'domainnet':
        K = 6
        num_nodes = K*client_group
        dataset_splited_list = [generate_DomainNet(client_group=client_group, method='iid', alpha=10)]
    else:
        K = 4
        num_nodes = 40 #40
        noise = None #'rotation'
        dataset_splited_list = [
            # data_process(dataset).split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'),\
            # data_process(dataset).split_dataset_groupwise(K, 3, 'class', int(num_nodes/K), 2, 'class'),\
            # data_process(dataset).split_dataset(num_nodes, 3, 'class'),\
            # data_process(dataset).split_dataset(num_nodes, 0.1, 'dirichlet'),\
            # data_process(dataset).split_dataset_groupwise(K, 10, 'dirichlet', int(num_nodes/K), 0.1, 'dirichlet', noise),\
            # data_process(dataset).split_dataset_groupwise(K, 10, 'dirichlet', int(num_nodes/K), 10, 'dirichlet', noise),\
            # data_process(dataset).split_dataset_groupwise(K, 5, 'class', int(num_nodes/K), 2, 'class', noise) ,\
            data_process(dataset).split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 10, 'dirichlet', noise)]
    
    n_assign_list = [3,6]
    
    model_name_list0 = ['FedEM' ]#,'FedSoft',,] #,，BayesPFedAvg,FedSoft'BayesFedAvg','Fesem',,,,,'GNN','FedAvg','FedAvg','Wecfl',Fesem
    model_name_list1 = ['GNN']
    model_name_list2 = ['JPDA','MHT'] #,'MHT'
    # cost_methods = ['weighted'] #,'average'
    K_set = K
    warm_ups = [False,True]
    Parallel(n_jobs=2)(delayed(main)(seeds, dataset_splited, model, model_name, K_set) \
                        for dataset_splited in dataset_splited_list \
                        for model_name in model_name_list0)
    
    # Parallel(n_jobs=2)(delayed(main)(seeds, dataset_splited, model, model_name, K_set, warm_up=warm_up) \
    #                     for dataset_splited in dataset_splited_list \
    #                     for model_name in model_name_list1 \
    #                         for warm_up in warm_ups)

    # Parallel(n_jobs=1)(delayed(main)(seeds, dataset_splited, model, model_name,K_set, n_assign, warm_up) \
    #                     for dataset_splited in dataset_splited_list \
    #                     for model_name in model_name_list2 \
    #                         for n_assign in n_assign_list \
    #                         for warm_up in warm_ups)

    
    # main(seeds, dataset_splited_list[0], model, model_name_list2[0], K=K, n_assign=n_assign_list[0],warm_up=warm_ups[0])
    
    # main(seeds, dataset_splited_list[0], model, model_name_list2[0], K=K, n_assign=n_assign_list[0],warm_up=warm_ups[1])

    # main(seeds, dataset_splited_list[0], model, model_name_list2[0], K=K, n_assign=n_assign_list[1],warm_up=warm_ups[0])

    # main(seeds, dataset_splited_list[0], model, model_name_list2[0], K=K, n_assign=n_assign_list[1],warm_up=warm_ups[1])

    # main(seeds, dataset_splited_list[0], model, model_name_list2[1], K=K, n_assign=n_assign_list[0],warm_up=warm_ups[0])
    
    # main(seeds, dataset_splited_list[0], model, model_name_list2[1], K=K, n_assign=n_assign_list[0],warm_up=warm_ups[1])

    # main(seeds, dataset_splited_list[0], model, model_name_list2[1], K=K, n_assign=n_assign_list[1],warm_up=warm_ups[0])

    # main(seeds, dataset_splited_list[0], model, model_name_list2[1], K=K, n_assign=n_assign_list[1],warm_up=warm_ups[1])
