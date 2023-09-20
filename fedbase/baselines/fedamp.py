'''
can be used for Bayesian or non-Bayesian, and IFCA or WECFL methods
assign_method=[”ifca”,'wecfl'], bayes=[True, False]
'''

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

def run(dataset_splited, batch_size, K, num_nodes, model, objective, optimizer, global_rounds, local_steps, \
    reg_lam = 1.0, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),\
         bayes=False):
    # dt = data_process(dataset)
    # train_splited, test_splited = dt.split_dataset(num_nodes, split['split_para'], split['split_method'])
    train_splited, test_splited, split_para = dataset_splited
    server = server_class(device)
    server.assign_model(model())
    model_lambda = dict()
    for name, param in server.model.named_parameters():
        model_lambda[name] = torch.ones_like(param)
    server.assign_model_lambda(model_lambda)

    nodes = [node(i, device) for i in range(num_nodes)]
    # local_models = [model() for i in range(num_nodes)]
    # local_loss = [objective() for i in range(num_nodes)]
    # bayes = True

    for i in range(num_nodes):
        # data
        # print(len(train_splited[i]), len(test_splited[i]))
        nodes[i].assign_train(DataLoader(train_splited[i], batch_size=batch_size, shuffle=True))
        nodes[i].assign_test(DataLoader(test_splited[i], batch_size=batch_size, shuffle=False))
        # model
        nodes[i].assign_model(model())
        nodes[i].assign_model_lambda(model_lambda)
        # objective
        nodes[i].assign_objective(objective())
        # optim
        nodes[i].assign_optim(optimizer(nodes[i].model.parameters()))
    
    del train_splited, test_splited

    # initialize parameters to nodes
    server.distribute([nodes[i].model for i in range(num_nodes)])
    weight_list = [nodes[i].data_size/sum([nodes[i].data_size for i in range(num_nodes)]) for i in range(num_nodes)]

    
    # train!
    for t in range(global_rounds):
        print('-------------------Global round %d start-------------------' % (t))
        # 1. local update
        for j in range(num_nodes):
            print('-------------------Local update %d start-------------------' % (j))
            # if not bayes:
            cluster_model = copy.deepcopy(nodes[j].model)
            nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_fedprox, reg_lam = reg_lam, reg_model = cluster_model))
            # else:
            #     nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_bayes, reg_model = nodes[j].model, reg_model_lambda = nodes[j].model_lambda, reg_lam = reg_lam))
            print('-------------------Local update %d end-------------------' % (j))

        # 2. server update
        # 2.1 model aggregation to K cluster by model parameters similarity
        # 2.1.1 get the model parameters
        model_list = [nodes[i].model for i in range(num_nodes)]
        # 2.1.2 get the model parameters similarity
        # cluster_models, assign_labels = server.fuzzy_clustering(model_list, K, weight_list=weight_list)
        cluster_models, assign_labels = server.aggregate_amp(model_list, weight_list=weight_list)
        # 2.2 send the close model to the corresponding node
        for i in range(num_nodes):
            for k in range(K):
                if assign_labels[i] == k:
                    nodes[i].assign_model(cluster_models[k])
            # server.distribute([nodes[i].model for i in range(num_nodes) if assign_labels[i] == k], cluster_models[k])
        # 3. accuracy evaluation
        for j in range(num_nodes):
            nodes[j].local_test()
        server.acc(nodes, weight_list)
        # delete the cluster_models
        del cluster_models
    # log
    log(os.path.basename(__file__)[:-3] +add_(K) + add_(reg_lam) + add_(split_para), nodes, server)

