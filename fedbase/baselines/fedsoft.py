
from fedbase.utils.data_loader import data_process, log
# from fedbase.nodes.node import node
from fedbase.nodes.node_fl_mot import node
from fedbase.utils.tools import add_
# from fedbase.server.server import server_class
from fedbase.server.server_fl_mot import server_class
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from fedbase.model.model import CNNCifar, CNNMnist
import os
import sys
import inspect
from functools import partial
import copy
import time
import numpy as np

device = torch.device("mps") # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def run(dataset_splited,batch_size,K,num_nodes,model,objective,optimizer,global_rounds,local_steps,selection_size,\
        num_classes,estimation_interval=1,reg_lam = None,do_selection=True,\
            device = device,accuracy_type='single',path='log/',finetune=False, finetune_steps = None):

    # 1. initialization
    train_splited, test_splited, split_para = dataset_splited
    server = server_class(device)
    server.assign_model(model())
    nodes = [node(i, device) for i in range(num_nodes)]

    for i in range(num_nodes):
        # data
        # print(len(train_splited[i]), len(test_splited[i]))
        nodes[i].assign_train(DataLoader(train_splited[i], batch_size=batch_size, shuffle=True))
        nodes[i].assign_test(DataLoader(test_splited[i], batch_size=batch_size, shuffle=False))
        # model
        nodes[i].assign_model(model())
        # objective
        nodes[i].assign_objective(objective())
        # optim
        nodes[i].assign_optim(optimizer(nodes[i].model.parameters()))
    
    del train_splited, test_splited

    # initialize parameters to nodes
    server.distribute([nodes[i].model for i in range(num_nodes)])
    weight_list = [nodes[i].data_size/sum([nodes[i].data_size for i in range(num_nodes)]) for i in range(num_nodes)]

    # initialize K cluster model
    cluster_models = [model() for i in range(K)]

    for t in range(global_rounds):
        print('-------------------Global round %d start-------------------' % (t))
        # 1. importance estimatin
        if t % estimation_interval == 0:
            importance_weights_matrix = [] # dim = (num_nodes, num_clusters)
            for j in range(num_nodes):
                nodes[j].estimate_importance_weights(cluster_models,num_classes,count_smoother=0.0001)
                importance_weights_matrix.append(nodes[j].get_importance())
            importance_weights_matrix = np.array(importance_weights_matrix)
            importance_weights_matrix /= np.sum(importance_weights_matrix, axis=0)
        
        # 2. Client selection
        selection = []
        if do_selection:
            for s in range(K):
                selection.append(np.random.choice(a=range(num_nodes), size=selection_size,
                                                    p=importance_weights_matrix[:, s], replace=False).tolist())
        else:
            selection = np.tile(range(num_nodes), reps=(K, 1))
        server.clustering['label'].append(selection)

        # 3. local update
        # Local updates
        for node_index in np.unique(np.concatenate(selection).ravel()):
            print('-------------------Local update %d start-------------------' % (node_index))
            nodes[node_index].local_update_steps(local_steps, partial(nodes[node_index].train_single_step_fedsoft, cluster_vec = cluster_models, reg_lam= reg_lam))
            
        # 4. server aggretation
        cluster_models = server.aggregate_fedsoft(selection, K,nodes, cluster_models, do_selection=True)

        # 5. test accuracy
        for j in range(num_nodes):
            nodes[j].local_test()
        server.acc(nodes,weight_list)

    if not finetune:
            # assign = [[i for i in range(num_nodes) if nodes[i].label == k] for k in range(K)]
            # log
            log(os.path.basename(__file__)[:-3] +add_(K) + add_(reg_lam)+ add_(split_para), nodes, server,path=path)
            return cluster_models
    else:
        if not finetune_steps:
            finetune_steps = local_steps
        # fine tune
        for j in range(num_nodes):
            if not reg_lam:
                nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_fedsoft, reg_model = cluster_models[nodes[j].label], reg_lam=reg_lam))
            else:
                nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_fedprox, reg_model = cluster_models[nodes[j].label], reg_lam= reg_lam))
            nodes[j].local_test()
        server.acc(nodes, weight_list)
        # log
        log(os.path.basename(__file__)[:-3] + add_('finetune') + add_(K) + add_(reg_lam) + add_(split_para), nodes, server)
        return [nodes[i].model for i in range(num_nodes)]




