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

@unpack_args 
def main0(seeds, dataset_splited, model):
    np.random.seed(seeds)
    central.run(dataset_splited, batch_size, model, nn.CrossEntropyLoss, optimizer, global_rounds, device = device)

@unpack_args
def main1(seeds, dataset_splited, model):
    np.random.seed(seeds)
    fedavg.run(dataset_splited, batch_size, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, device = device)
    # fedavg_finetune.run(dataset_splited, batch_size, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 10, device = device)
    # local.run(dataset_splited, batch_size, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, device = device)
    # ditto.run(dataset_splited, batch_size, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 0.95, device = device)
    fedprox.run(dataset_splited, batch_size, num_nodes, model,  nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 0.1, device = device)

@unpack_args
def main2(seeds, dataset_splited, model, K):
    np.random.seed(seeds)
    fedavg_ensemble.run(dataset_splited, batch_size, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, K, device = device)
    fedprox_ensemble.run(dataset_splited, batch_size, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, K, device = device)
    # wecfl.run(dataset_splited, batch_size, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, device = device)
    # fesem.run(dataset_splited, batch_size, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, device = device)
    # ifca.run(dataset_splited, batch_size, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, device = device)
    # wecfl.run(dataset_splited, batch_size, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 0.95, device = device)
    # fesem.run(dataset_splited, batch_size, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 0.95, device = device)
    # ifca.run(dataset_splited, batch_size, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 0.95, device = device)
    # fed_mot.run(dataset_splited, batch_size, K, num_nodes, model, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, reduction= 'GNN',device = device)

# multiprocessing
if __name__ == '__main__':
    seed = 1024
    np.random.seed(seed)
    torch.manual_seed(seed)
    # data_process('cifar10').split_dataset(200,2,'class')
    # for i in range(3):
    #     data_process('cifar10').split_dataset_groupwise(10,0.1,'dirichlet',20,10,'dirichlet')
    # for i in range(1):
    #     np.random.seed(i)
    # data_process('fashion_mnist').split_dataset_groupwise(10,0.1,'dirichlet',20,10,'dirichlet', plot_show=True)
    # data_process('fashion_mnist').split_dataset_groupwise(10,3,'class',20,2,'class', plot_show=True)
    # data_process('medmnist_pathmnist').split_dataset_groupwise(10,0.1,'dirichlet',20,10,'dirichlet', plot_show=True)
    # data_process('medmnist_octmnist').split_dataset_groupwise(10,0.1,'dirichlet',20,10,'dirichlet', plot_show=True)
    # data_process('medmnist_tissuemnist').split_dataset_groupwise(10,3,'class',20,2,'class', plot_show=True)
    # data_process('fashion_mnist').split_dataset(200,0.1,'dirichlet', plot_show= True)
    # data_process('fashion_mnist').split_dataset(200,2,'class', plot_show= True)
    # print(a)
    # # data_process('fashion_mnist').split_dataset_groupwise(10,3,'class',20, 2,'class', plot_show=True)
    # # data_process('fashion_mnist').split_dataset(18,0.1,'dirichlet', plot_show= True)
    # # data_process('cifar10').split_dataset(200,2,'class', plot_show= True)
    # # data_process('medmnist_octmnist').split_dataset(200,2,'class', plot_show= True)
    # print(a)
    # data_process('medmnist_pathmnist').split_dataset(200,2,'class', plot_show= True)
    # ditto.run(data_process('fashion_mnist').split_dataset_groupwise(5,6,'class',10,5,'class'), 16, 10, CNNFashion_Mnist,  nn.CrossEntropyLoss, optimizer, 3, 10, 0.95)
    # fedprox.run(data_process('fashion_mnist').split_dataset_groupwise(10,6,'class',20,5,'class'), batch_size, num_nodes, CNNFashion_Mnist,  nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 1)
    # fedprox_ensemble.run(data_process('fashion_mnist').split_dataset_groupwise(10,6,'class',20,5,'class'), 16,10, CNNFashion_Mnist,  nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, 1, 3)
    # fedavg.run(data_process('fashion_mnist').split_dataset_groupwise(5,6,'class',10,5,'class'), 16, 10, CNNFashion_Mnist,  nn.CrossEntropyLoss, optimizer, 3, 10,finetune=True)
    # fedavg_ensemble.run(data_process('fashion_mnist').split_dataset_groupwise(5,6,'class',10,5,'class'), 16,10, CNNFashion_Mnist,  nn.CrossEntropyLoss, optimizer, 2, 10, 3)
    # ifca.run(data_process('cifar10').split_dataset_groupwise(10,0.1,'dirichlet',20,10,'dirichlet'), batch_size, 10, num_nodes, CNNCifar, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # wecfl.run(data_process('medmnist_tissuemnist').split_dataset_groupwise(10,0.1,'dirichlet',20,10,'dirichlet'), batch_size, 10, num_nodes, CNNTissue, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, reg=0)
    # fesem.run(data_process('cifar10').split_dataset_groupwise(10, 0.1, 'dirichlet', 20, 10, 'dirichlet'), batch_size, 10, num_nodes, CNNCifar, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, reg_lam=0.001, finetune=True)
    # wecfl.run(data_process('cifar10').split_dataset_groupwise(5, 3, 'class', 40, 2, 'class'), batch_size, 5, num_nodes, CNNCifar, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # cfl_res.run(data_process('cifar10').split_dataset_groupwise(5, 3, 'class', 40, 2, 'class'), batch_size, 5, num_nodes, CNNCifar, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # fesem_cam.run(data_process('fashion_mnist').split_dataset(200, 2, 'class'), batch_size, 5, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optimizer, 2, global_rounds, local_steps, finetune =True, reg_lam=0.01)
    # wecfl_con.run(data_process('fashion_mnist').split_dataset(200, 2, 'class'), batch_size, 5, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, warmup_rounds = 1, tmp = 0.1, mu =1, base = 'parameter', reg_lam = 0.01)
    # fesem_con.run(data_process('fashion_mnist').split_dataset(200, 2, 'class'), batch_size, 5, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, warmup_rounds = 1, tmp = 0.1, mu =1, base = 'representation', reg_lam = 0.01)
    # wecfl_con.run(data_process('fashion_mnist').split_dataset(200, 2, 'class'), batch_size, 5, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, warmup_rounds = 1, tmp = 0.1, mu =10, base = 'parameter', reg_lam = 0.01)
    # ifca_con.run(data_process('fashion_mnist').split_dataset(200, 2, 'class'), batch_size, 5, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, warmup_rounds = 1, tmp = 0.1, mu =1, base = 'parameter', reg_lam = 0.01)
    # wecfl.run(data_process('cifar10').split_dataset(200, 0.1, 'dirichlet'), batch_size, 5, num_nodes, CNNCifar, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # wecfl_res.run(data_process('cifar10').split_dataset_groupwise(5, 3, 'class', 40, 2, 'class'), batch_size, 5, num_nodes, CNNCifar, nn.CrossEntropyLoss, optimizer, 2, global_rounds, local_steps)
    # ifca.run(data_process('fashion_mnist').split_dataset_groupwise(10, 0.1, 'dirichlet', 20, 5, 'dirichlet'), batch_size, 10, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # ifca.run(data_process('cifar10').split_dataset_groupwise(10, 0.1, 'dirichlet', 20, 5, 'dirichlet'), batch_size, 10, num_nodes, CNNCifar, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # ifca.run(data_process('medmnist_octmnist').split_dataset_groupwise(10, 0.1, 'dirichlet', 20, 5, 'dirichlet', plot_show= True), batch_size, 10, num_nodes, oct_net, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # print(a)
    ## Fashion MNIST
    # fedavg.run(data_process('fashion_mnist').split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'), batch_size,num_nodes, CNNFashion_Mnist,  nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # fedavg_bayes.run(data_process('fashion_mnist').split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'), batch_size,num_nodes, CNNFashion_Mnist,  nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # fedavg_ensemble.run(data_process('fashion_mnist').split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'), batch_size,num_nodes, CNNFashion_Mnist,  nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, K)
    # fedavg_bayes.run(data_process('fashion_mnist').split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'), batch_size,num_nodes, CNNFashion_Mnist,  nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, weight_method = 'loss',aggregated_method='AA')
    # fed_mot_ensemble.run(data_process('fashion_mnist').split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'), batch_size,num_nodes, CNNFashion_Mnist,  nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, K)
    # fed_mot_ifca.run(data_process('fashion_mnist').split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'), batch_size,K, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # ifca.run(data_process('fashion_mnist').split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'), batch_size,K, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # wecfl.run(data_process('fashion_mnist').split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'), batch_size, K, num_nodes, CNNFashion_Mnist, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    ## Mnist
    # fedavg.run(data_process('mnist').split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'), batch_size,num_nodes, CNNMnist,  nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # DATA SIZE AND GA
    # fedavg_bayes.run(data_process('mnist').split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'), batch_size,num_nodes, CNNMnist,  nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, device, weight_method = 'data_size',aggregated_method='GA')
    # LOSS AND AA
    # fedavg_bayes.run(data_process('mnist').split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'), batch_size,num_nodes, CNNMnist,  nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, device, weight_method = 'loss',aggregated_method='AA')

    # fedavg_ensemble.run(data_process('mnist').split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'), batch_size,num_nodes, CNNMnist,  nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, K)
    # K = 2
    fed_mot_ensemble.run(data_process('mnist').split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'), batch_size, num_nodes, CNNMnist,  nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, K, device=device, weight_type='loss')
    # fed_mot_GNN.run(data_process('mnist').split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'), batch_size,K, num_nodes, CNNMnist, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps,assign_method='wecfl',bayes=True)
    # assign_method='wecfl',bayes=false
    # fed_mot_GNN.run(data_process('mnist').split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'), batch_size,K, num_nodes, CNNMnist, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps,assign_method='wecfl',bayes=False,device=device)
    # assign_method='ifca',bayes=True
    # fed_mot_GNN.run(data_process('mnist').split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'), batch_size,K, num_nodes, CNNMnist, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps,assign_method='ifca',bayes=True,device=device)
    # # MHT:ifca, GA
    # single_MHT2.run(data_process('mnist').split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'), batch_size,K, num_nodes, CNNMnist, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps,assign_method='ifca',bayes=True,weight_type='loss',device=device)

    # ifca.run(data_process('mnist').split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'), batch_size,K, num_nodes, CNNMnist, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # wecfl.run(data_process('mnist').split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'), batch_size, K, num_nodes, CNNMnist, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    # wecfl_bayes.run(data_process('mnist').split_dataset_groupwise(K, 0.1, 'dirichlet', int(num_nodes/K), 5, 'dirichlet'), batch_size, K, num_nodes, CNNMnist, nn.CrossEntropyLoss, optimizer, global_rounds, local_steps)
    
    # for MHT
    # K = 1
    # H = 2
    # fed_mot_MHT.run(data_process('mnist').split_dataset(num_nodes, 2, 'class'), batch_size,num_nodes, CNNMnist,  nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, H, device, n_ensemble=1)
    # fedavg_bayes.run(data_process('mnist').split_dataset(num_nodes, 2, 'class'), batch_size,num_nodes, CNNMnist,  nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, device, weight_method = 'loss',aggregated_method='AA')
    # single_MHT2.run(data_process('mnist').split_dataset(num_nodes, 2, 'class'), batch_size,num_nodes, CNNMnist,  nn.CrossEntropyLoss, optimizer, global_rounds, local_steps, H, device)

    # multi_processes = 2
    # seeds = 1
    # # Run
    # start = time.perf_counter()
    # mp.set_start_method('spawn')
    # with mp.Pool(multi_processes) as p:
    #     # group_wise
    #     # p.map(main4, [(i, data_process(dataset).split_dataset_groupwise(n0,j0,k0,n1,j1,k1), model) for i in range(27, 27+seeds) for dataset, model in zip(['cifar10', 'fashion_mnist'],[CNNCifar, CNNFashion_Mnist]) \
    #     #     for n0,n1 in zip([5, 10],[40, 20]) for j0, k0, j1, k1 in zip([6, 0.1], ['class', 'dirichlet'], [5, 10], ['class', 'dirichlet'])])
    #     # p.map(main5, [(i, data_process(dataset).split_dataset_groupwise(n0,j0,k0,n1,j1,k1), model, K) for i in range(27, 27+seeds) for dataset, model in zip(['cifar10', 'fashion_mnist'],[CNNCifar, CNNFashion_Mnist]) \
    #     # for K,n0,n1 in zip([5, 10], [5, 10],[40, 20]) for j0, k0, j1, k1 in zip([6, 0.1], ['class', 'dirichlet'], [5, 10], ['class', 'dirichlet'])])
    #     # client_wise
    #     # p.map(main1, [(i, data_process(dataset).split_dataset(num_nodes, j, k), model) for i in range(27, 27+seeds) for dataset, model in zip(['cifar10', 'fashion_mnist'],[CNNCifar, CNNFashion_Mnist]) for j, k in zip([2, 0.1], ['class', 'dirichlet'])])
    #     # p.map(main2, [(i, data_process(dataset).split_dataset(num_nodes, j, k), model, K) for i in range(27, 27+seeds) for dataset, model in zip(['medmnist_octmnist'],[oct_net]) for j, k in zip([2, 0.1], ['class', 'dirichlet']) for K in [3,5,10]])
    #     p.map(main2, [(i, data_process(dataset).split_dataset(num_nodes, j, k), model, K) for i in range(27, 27+seeds) for dataset, model in zip(['fashion_mnist'],[CNNFashion_Mnist]) for j, k in zip([2, 0.1], ['class', 'dirichlet']) for K in [3]])

    #     p.close()

    # print(time.perf_counter()-start, "seconds")