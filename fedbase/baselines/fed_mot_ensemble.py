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

def run(dataset_splited, batch_size, num_nodes, model, objective, optimizer, global_rounds, local_steps, n_ensemble, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),weight_type='loss'):
    # dt = data_process(dataset)
    # train_splited, test_splited = dt.split_dataset(num_nodes, split['split_para'], split['split_method'])
    train_splited, test_splited, split_para = dataset_splited
    print('data splited')
    server = server_class(device)
    server.assign_model(model())
    model_lambda = dict()
    for name, param in model().named_parameters():
        model_lambda[name] = torch.ones_like(param)
    server.assign_model_lambda(model_lambda)
    
    # nodes = [node(i, device) for i in range(num_nodes)]
    # this method can be defined outside your model class
    # def weights_init(m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.normal_(m.weight, mean=0.0, std=1.0)
    cluster_models = [model() for i in range(n_ensemble)]
    cluster_models_lambda = [model_lambda for i in range(n_ensemble)]

    # servers_list = []
    nodes_list = []
    for n_en in range(n_ensemble):
        # server = server_class(device)
        # server.assign_model(model())
        # model_lambda = dict()
        # for name, param in server.model.named_parameters():
        #     model_lambda[name] = torch.ones_like(param)
        # server.assign_model_lambda(model_lambda)
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
            nodes[i].assign_model_lambda(model_lambda)
            # objective
            nodes[i].assign_objective(objective())
            # optim
            nodes[i].assign_optim(optimizer(nodes[i].model.parameters()))
        # servers_list.append(server)
        nodes_list.append(nodes)
        print('K cluster %d initialized' % (n_en))

    # initialize parameters to nodes
    weight_list = [nodes[i].data_size/sum([nodes[i].data_size for i in range(num_nodes)]) for i in range(num_nodes)]
    server.distribute([nodes[i].model for i in range(num_nodes)])

    # train!
    for round in range(global_rounds):
        print('-------------------Global round %d start-------------------' % (round))
        # single-processing!
        for k in range(n_ensemble):
            # server = servers_list[k]
            nodes = nodes_list[k]
            # sample the nodes
            # nodes = server.sample_nodes(nodes_ori, sampling_rate=0.8, sample_with_replacement=False)
            nodes_weight = []
            for j in range(num_nodes):
                # nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step))
                ce_loss = nodes[j].get_ce_loss()
                nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_bayes, reg_model = cluster_models[n_en], reg_model_lambda = cluster_models_lambda[n_en]))
                nodes_weight.append(nodes[j].weight)
                # server aggregation weight
            nodes_weight = torch.tensor(nodes_weight)
            if weight_type == 'loss':
                nodes_weight = nodes_weight/nodes_weight.sum(dim=0)
            elif weight_type == 'similarity':
                pass
            # print('nodes_weight', nodes_weight)
            nodes_weight = torch.tensor(nodes_weight)
            # randomly remove some nodes from the aggregation
            for j in range(num_nodes):
                if round <=5 and nodes_weight[j] < 0.1:
                    nodes_weight[j] = nodes_weight[j]*torch.tensor([1 if torch.rand(1) > 0.3 else 0])
            nodes_weight = nodes_weight/nodes_weight.sum(dim=0)
            print('nodes_weight', nodes_weight)
            # server aggregation and distribution
            model_k, model_k_lambda,_ = server.aggregate_bayes([nodes[i].model for i in range(num_nodes)],\
                    [nodes[i].model_lambda for i in range(num_nodes)], nodes_weight, aggregated_method='AA')
            server.distribute([nodes[i].model for i in range(num_nodes)],model_k)
            server.distribute_lambda([nodes[i].model_lambda for i in range(num_nodes)], model_k_lambda)

            # test accuracy
            # for j in range(num_nodes):
            #     nodes[j].local_test()
            # server.acc(nodes, weight_list)

            # update the cluster model
            for name, param in cluster_models[k].named_parameters():
                cluster_models[k].state_dict()[name].data.copy_(model_k[name])
                cluster_models_lambda[k][name].data.copy_(model_k_lambda[name])
            
            # update the servers_list and nodes_list
            # servers_list[k] = server
            nodes_list[k] = nodes

        # print the ensemble accuracy
        print('test ensemble\n')
        for j in range(num_nodes):
            nodes[j].local_ensemble_test(cluster_models, voting = 'soft')
        server.acc(nodes, weight_list)


    # test ensemble
    # print('test ensemble\n')
    # for j in range(num_nodes):
    #     nodes[j].local_ensemble_test(cluster_models, voting = 'soft')
    # server.acc(nodes, weight_list)

    # log
    log(os.path.basename(__file__)[:-3] + add_(n_ensemble) + add_(split_para), nodes, server)

    return cluster_models