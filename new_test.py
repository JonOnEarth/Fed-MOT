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


os.chdir(os.path.dirname(os.path.abspath(__file__))) # set the current path as the working directory
global_rounds = 50
num_nodes = 10
local_steps = 10
batch_size = 32
optimizer = partial(optim.SGD,lr=0.01, momentum=0.9)
# optimizer = partial(optim.SGD,lr=0.001)
# device = torch.device('cuda:2')
# device = torch.device('cuda')  # Use GPU if available
device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
# print(device)
K = 2
H = 2

def main(seeds, dataset_splited, model, model_name, K=None,n_assign=None):
    np.random.seed(seeds)
    # dataset_splited, model = dataset_splited_model[0], dataset_splited_model[1]
    if model_name == 'FedAvg':
        fedavg.run(dataset_splited, batch_size, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, device = device)
    elif model_name == 'BayesFedAvg':
        fedavg_bayes.run(dataset_splited,batch_size, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, device = device)
    elif model_name == 'Wecfl':
        fed_mot_GNN.run(dataset_splited,batch_size,K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, assign_method='wecfl',bayes=False, device = device)
    elif model_name == 'Fesem':
        fesem.run(dataset_splited,batch_size, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, reg_lam=0.001, device = device)
    elif model_name == 'GNN':
        fed_mot_GNN.run(dataset_splited,batch_size,K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, assign_method='ifca',bayes=True, device = device)
    elif model_name == 'JPDA':
        fed_mot_jpda.run(dataset_splited, batch_size, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, bayes=True, num_assign=n_assign,device=device)
    elif model_name == 'MHT':
        fed_mot_mht.run(dataset_splited, batch_size, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, bayes=True, num_assign=n_assign,hypothesis=n_assign, device=device)

if __name__ == '__main__':
    dataset = 'mnist'
    seeds = 0
    if dataset == 'mnist':
        model = CNNMnist
    elif dataset == 'femnist':
        model = CNNFemnist
    elif dataset == 'fashion_mnist':
        model = CNNFashion_Mnist
    elif dataset == 'cifar10':
        model = CNNCifar

    K = 4
    num_nodes = 20
    n_assign_list = [2,4,6]
    noise = 'rotation'
    dataset_splited_list = [
        data_process(dataset).split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'),\
        data_process(dataset).split_dataset_groupwise(K, 3, 'class', int(num_nodes/K), 2, 'class')#,\
        # data_process(dataset).split_dataset(num_nodes, 2, 'class'),\
        # data_process(dataset).split_dataset(num_nodes, 0.1, 'dirichlet'),\
        # data_process(dataset).split_dataset_groupwise(K, 10, 'dirichlet', int(num_nodes/K), 0.1, 'dirichlet', noise),\
        # data_process(dataset).split_dataset_groupwise(K, 10, 'dirichlet', int(num_nodes/K), 10, 'dirichlet', noise)
                        ] # rotation
    model_name_list1 = ['FedAvg','Wecfl','GNN'] #'BayesFedAvg','Fesem',
    model_name_list2 = ['JPDA'] #,'MHT'

    Parallel(n_jobs=8)(delayed(main)(seeds, dataset_splited, model, model_name, K+1) \
                        for dataset_splited in dataset_splited_list \
                        for model_name in model_name_list1)
    
    Parallel(n_jobs=2)(delayed(main)(seeds, dataset_splited, model, model_name,K+1, n_assign) \
                        for dataset_splited in dataset_splited_list \
                        for model_name in model_name_list2 \
                            for n_assign in n_assign_list)

    
    # main(seeds, dataset_splited_list[-1], model, model_name_list2[0], K=K, n_assign=n_assign_list[1])
    # main(seeds, dataset_splited_list[-1], model, model_name_list2[1], K=K, n_assign=n_assign_list[1])

