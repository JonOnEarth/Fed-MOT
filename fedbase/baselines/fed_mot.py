from fedbase.utils.data_loader import data_process, log
from fedbase.nodes.node_fl_mot import node
from fedbase.utils.tools import add_
from fedbase.server.server_fl_mot import server_class
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from fedbase.model.model import CNNCifar, CNNMnist
import os
import sys
import inspect
from functools import partial

def run(dataset_splited, batch_size, K, num_nodes, model, objective, optimizer, global_rounds, local_steps, \
    reduction = None, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), finetune=False, finetune_steps = None, temperature=1.0):
    # dt = data_process(dataset)
    # train_splited, test_splited = dt.split_dataset(num_nodes, split['split_para'], split['split_method'])
    train_splited, test_splited, split_para = dataset_splited
    server = server_class(device)
    server.assign_model(model())
    # define the parameter lambda
    model_lambda = dict()
    for name, param in model.named_parameters():
        model_lambda[name] = torch.ones_like(param.data).to(device)
    server.assign_lambda(model_lambda)

    nodes = [node(i, device) for i in range(num_nodes)]
    # local_models = [model() for i in range(num_nodes)]
    # local_loss = [objective() for i in range(num_nodes)]

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
    # weight_list = [nodes[i].data_size/sum([nodes[i].data_size for i in range(num_nodes)]) for i in range(num_nodes)]
    weight_list = [1.0 for i in range(num_nodes)]

    # initialize K cluster model
    cluster_models = [model() for i in range(K)]

    # train!
    for t in range(global_rounds):
        print('-------------------Global round %d start-------------------' % (t))
        
        # local update
        nodes_k_m = [[] for j in range(num_nodes)]
        nodes_k_m_weight = [[] for j in range(num_nodes)]
        for j in range(num_nodes):
            nodes_k = [[] for i in range(K)]
            nodes_k_weight = [[] for i in range(K)]
            for i in range(K):
                nodes[j].get_ce_loss(temperature)
                if reduction == 'JPDA':
                    nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_bayes))
                elif reduction == 'GNN':
                    continue
                nodes_k[i].append(nodes[j])
                nodes_k_weight[i].append(nodes[j].weight)
            nodes_k_m[j] = nodes_k
            nodes_k_m_weight[j] = nodes_k_weight
        # server aggregation and distribution by cluster
        if reduction == 'JPDA':
            for j in range(K):
                model_k, model_k_lambda = server.aggregate_bayes([nodes_k_m[i][j][0].model for i in range(num_nodes)],\
                                                  [nodes_k_m_weight[i][j][0] for i in range(num_nodes)],aggregated_method='AA')

        # server clustering
        # server.weighted_clustering(nodes, list(range(num_nodes)), K)
        # server aggregation and distribution by cluster
        for j in range(K):
            assign_ls = [i for i in list(range(num_nodes)) if nodes[i].label==j]
            weight_ls = [nodes[i].data_size/sum([nodes[i].data_size for i in assign_ls]) for i in assign_ls]
            model_k = server.aggregate([nodes[i].model for i in assign_ls], weight_ls)
            server.distribute([nodes[i].model for i in assign_ls], model_k)
            cluster_models[j].load_state_dict(model_k)

        # test accuracy
        for j in range(num_nodes):
            nodes[j].local_test()
        server.acc(nodes, weight_list)