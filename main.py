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
# import multiprocessing as mp
# import time
import torchvision.models as models

# from joblib import Parallel, delayed
import argparse
from fedbase.utils.get_digit5 import generate_Digit5
from fedbase.utils.get_amazon_review import generate_AmazonReview
from fedbase.utils.get_domainnet import generate_DomainNet

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main(args):
    np.random.seed(args.seeds)
    if args.model_name == 'FedAvg':
        fedavg.run(args.dataset_splited, args.batch_size, args.num_nodes, args.model, nn.CrossEntropyLoss, args.optimizer, args.global_rounds, args.local_steps, device = args.device, path=args.path)
    elif args.model_name == 'BayesFedAvg':
        fedavg_bayes.run(args.dataset_splited, args.batch_size, args.num_nodes, args.model, nn.CrossEntropyLoss, args.optimizer, args.global_rounds, args.local_steps, device = args.device,path=args.path)
    elif args.model_name == 'Wecfl':
        GNN.run(args.dataset_splited, args.batch_size, args.K, args.num_nodes, args.model, nn.CrossEntropyLoss, args.optimizer, args.global_rounds, args.local_steps, assign_method='wecfl',bayes=False, device = args.device,path=args.path)
    elif args.model_name == 'Fesem':
        fesem.run(args.dataset_splited, args.batch_size, args.K, args.num_nodes, args.model, nn.CrossEntropyLoss, args.optimizer, args.global_rounds, args.local_steps, reg_lam=0.001, device = args.device,path=args.path)
    elif args.model_name == 'GNN':
        GNN.run(args.dataset_splited, args.batch_size, args.K, args.num_nodes, args.model, nn.CrossEntropyLoss, args.optimizer, args.global_rounds, args.local_steps, assign_method='ifca',bayes=True, device = args.device,warm_up=args.warm_up,path=args.path)
    elif args.model_name == 'IFCA':
        GNN.run(args.dataset_splited, args.batch_size, args.K, args.num_nodes, args.model, nn.CrossEntropyLoss, args.optimizer, args.global_rounds, args.local_steps, assign_method='ifca',bayes=False, device = args.device,warm_up=args.warm_up,path=args.path)
    elif args.model_name == 'JPDA':
        jpda.run(args.dataset_splited, args.batch_size, args.K, args.num_nodes, args.model, nn.CrossEntropyLoss, args.optimizer, args.global_rounds, args.local_steps, bayes=True, num_assign=args.n_assign,device=args.device, cost_method=args.cost_method,warm_up=args.warm_up,path=args.path)
    elif args.model_name == 'MHT':
        mht.run(args.dataset_splited, args.batch_size, args.K, args.num_nodes, args.model, nn.CrossEntropyLoss, args.optimizer, args.global_rounds, args.local_steps, bayes=True, num_assign=args.n_assign,hypothesis=args.n_assign, device=args.device, cost_method=args.cost_method,warm_up=args.warm_up,path=args.path)
    elif args.model_name == 'Wecfl_unknown':
        GNN.run(args.dataset_splited, args.batch_size, args.K, args.num_nodes, args.model, nn.CrossEntropyLoss, args.optimizer, args.global_rounds, args.local_steps, assign_method='wecfl_unknown',bayes=False, device = args.device,path=args.path)
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='amazon', help='dataset')
    parser.add_argument('--seeds', type=int, default=1989, help='random seeds')
    parser.add_argument('--model_name', type=str, default='IFCA', help='model name')
    parser.add_argument('--num_nodes', type=int, default=10, help='number of nodes')
    parser.add_argument('--local_steps', type=int, default=10, help='number of local steps')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer')
    parser.add_argument('--global_rounds', type=int, default=100, help='global rounds')
    parser.add_argument('--K', type=int, default=7, help='number of clusters')
    parser.add_argument('--n_assign', type=int, default=3, help='number of assignments for JPDA and MHT')
    parser.add_argument('--cost_method', type=str, default='weighted', help='cost method')
    # parser.add_argument('--warm_up', type=bool, default=False, help='warm up')
    # Use the function in your argument parser
    # parser.add_argument('--warm_up', type=str2bool,default='0', help='Whether to warm up')
    parser.add_argument('--warm_up',type=str, default='False', help='Whether to warm up')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--client_group', type=int, default=2, help='client group for amazon and digit5')
    parser.add_argument('--path', type=str, default='log_new/', help='path')
    # parser.add_argument('--k_known', type=str, default='False', help='k_known')
    args = parser.parse_args()

    if args.dataset == 'mnist':
        args.model = CNNMnist
    elif args.dataset == 'femnist':
        args.model = CNNFemnist
    elif args.dataset == 'fashion_mnist':
        args.model = CNNFashion_Mnist
    elif args.dataset == 'cifar10':
        args.model = CNNCifar
    elif args.dataset == 'amazon':
        args.model = AmazonMLP
    elif args.dataset == 'domainnet':
        args.model = AlexNet
    elif args.dataset == 'digit5':
        args.model = Digit5CNN
    
    if args.optimizer == 'SGD':
        args.optimizer = partial(optim.SGD,lr=0.005, momentum=0.9)
    elif args.optimizer == 'Adam':
        args.optimizer = partial(optim.Adam,lr=0.001)
    
    if args.dataset == 'digit5':
        domains=['mnistm', 'mnist', 'usps', 'svhn','syn'] #,
        K = len(domains)
        args.num_nodes = K*args.client_group #40
        args.dataset_splited = generate_Digit5(domains=domains, client_group=args.client_group, method='iid', alpha=10)
    elif args.dataset == 'amazon':
        K = 4
        args.num_nodes = K*args.client_group
        args.dataset_splited = generate_AmazonReview(client_group=args.client_group, method='dirichlet', alpha=1)
    elif args.dataset == 'domainnet':
        K = 6
        args.num_nodes = K*args.client_group
        args.dataset_splited = generate_DomainNet(client_group=args.client_group, method='iid', alpha=10)
    elif args.dataset == 'mnist' or args.dataset == 'femnist' or args.dataset == 'fashion_mnist' or args.dataset == 'cifar10':
        # you can have more details setting for different non-iid setting, such as iid, dirichlet, class, etc.
        K = 4
        args.num_nodes = 40 #K*args.client_group
        args.dataset_splited = data_process(args.dataset).split_dataset_groupwise(K, 0.1, 'dirichlet', int(args.num_nodes/K), 10, 'dirichlet')

    if args.device == 'cuda':
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    main(args)
