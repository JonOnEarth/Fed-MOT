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
import copy
import torch.nn as nn

def run(dataset_splited, batch_size, num_nodes, model, objective, optimizer, global_rounds, local_steps, reg, n_ensemble, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # dt = data_process(dataset)
    # train_splited, test_splited = dt.split_dataset(num_nodes, split['split_para'], split['split_method'])
    train_splited, test_splited, split_para = dataset_splited
    print('data splited')

    models = []
    for _ in range(n_ensemble):
        server = server_class(device)
        server.assign_model(model())

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

        # initialize parameters to nodes
        weight_list = [nodes[i].data_size/sum([nodes[i].data_size for i in range(num_nodes)]) for i in range(num_nodes)]
        server.distribute([nodes[i].model for i in range(num_nodes)])