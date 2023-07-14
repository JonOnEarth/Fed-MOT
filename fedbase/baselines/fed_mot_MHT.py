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

def run(dataset_splited, batch_size, num_nodes, model, objective, optimizer, global_rounds, local_steps, n_ensemble, H, device,\
        weight_type='cluster'): #weight_type='data_size'
    # dt = data_process(dataset)
    # train_splited, test_splited = dt.split_dataset(num_nodes, split['split_para'], split['split_method'])
    train_splited, test_splited, split_para = dataset_splited
    server = server_class(device)
    server.assign_model(model())
    model_lambda = dict()
    for name, param in server.model.named_parameters():
        model_lambda[name] = torch.ones_like(param)
    server.assign_model_lambda(model_lambda)

    # this method can be defined outside your model class
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=1.0)
    # cluster_models = [[model().apply(weights_init).to(device) for i in range(n_ensemble)] for h in range(H)]
    cluster_models = [[model().to(device) for i in range(n_ensemble)] for h in range(H)]
    cluster_models_lambda = [[model_lambda for i in range(n_ensemble)] for h in range(H)]
    cluster_weights = [[1/H for i in range(n_ensemble)] for h in range(H)]
    # cluster_weights = torch.tensor(cluster_weights, device=device)
    
    nodes_list = []
    # for n_en in range(n_ensemble):
    #     nodes_H = []
    #     for h in range(H):
    #         nodes = [node(i, device) for i in range(num_nodes)]
    #         for i in range(num_nodes):
    #             # data
    #             # print(len(train_splited[i]), len(test_splited[i]))
    #             nodes[i].assign_train(DataLoader(train_splited[i], batch_size=batch_size, shuffle=True))
    #             nodes[i].assign_test(DataLoader(test_splited[i], batch_size=batch_size, shuffle=False))
    #             # model
    #             nodes[i].assign_model(model())
    #             nodes[i].assign_model_lambda(model_lambda)
    #             # objective
    #             nodes[i].assign_objective(objective())
    #             # optim
    #             nodes[i].assign_optim(optimizer(nodes[i].model.parameters()))
    #         # align the paramters of nodes
    #         server.distribute([nodes[i].model for i in range(num_nodes)])
    #         server.distribute_lambda([nodes[i].model_lambda for i in range(num_nodes)])
    #         nodes_H.append(nodes)
    #     nodes_list.append(nodes_H)
    #     print('K cluster %d initialized' % (n_en))
    for n_en in range(n_ensemble):
        nodes = [node(i, device) for i in range(num_nodes)]
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

    del train_splited, test_splited
    weight_list = [nodes[i].data_size/sum([nodes[i].data_size for i in range(num_nodes)]) for i in range(num_nodes)]
    
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

            # decide which nodes should cluster together
            if cluster_method == 'ifca':
                # local update
                assignment = [[] for i in range(n_ensemble)]
                for j in range(num_nodes):
                    m = 0
                    for h in range(1, H):
                        # print(nodes[i].local_train_loss(cluster_models[m]), nodes[i].local_train_loss(cluster_models[k]))
                        if nodes[j].local_train_loss(cluster_models[m][k])>=nodes[j].local_train_loss(cluster_models[h][k]):
                            m = h
                    assignment[m].append(j)
                    nodes[j].label = m
                server.clustering['label'].append(assignment)
            elif cluster_method == 'kmeans':
                server.weighted_clustering(nodes, list(range(num_nodes)), H, weight_type='loss')

            # aggregate the models
            models_H = [None for h in range(H)]
            models_H_lambda = [None for h in range(H)]
            weight_H = [None for h in range(H)]
            for h in range(H):
                assign_ls = [i for i in list(range(num_nodes)) if nodes[i].label==h]
                if weight_type == 'data_size':
                    weight_ls = [nodes[i].data_size/sum([nodes[i].data_size for i in assign_ls]) for i in assign_ls]
                else:
                    weight_ls = [nodes[i].weight for i in assign_ls]
                model_k, model_k_lambda, weight_h = server.aggregate_bayes([nodes[i].model for i in range(num_nodes)],\
                    [nodes[i].model_lambda for i in range(num_nodes)], nodes_weight, aggregated_method='AA')
                models_H[h] = model_k
                models_H_lambda[h] = model_k_lambda
                weight_H[h] = weight_h
                # send to local to calculate the loss
                server.distribute([nodes[i].model for i in list(range(num_nodes))], model_k)
                server.distribute_lambda([nodes[i].model_lambda for i in list(range(num_nodes))], model_k_lambda)
                if personalized_weight_type == 'equal':
                    for j in range(num_nodes):
                        nodes[j].weight = 1/H
                elif personalized_weight_type == 'loss':  
                    for j in range(num_nodes):
                        nodes[j].get_ce_loss()
                weight_h = 
            
            # test accuracy of each hypothesis of each cluster
            # for j in range(num_nodes):
            #     nodes[j].local_test()
            # server.acc(nodes, weight_list)

            # update the cluster model
            for h in range(H):
                server.distribute([nodes_list[n_en][h][i].model for i in range(num_nodes)],models_H[h])
                server.distribute_lambda([nodes_list[n_en][h][i].model_lambda for i in range(num_nodes)], models_H_lambda[h])

                for name, param in cluster_models[h][n_en].named_parameters():
                    cluster_models[h][n_en].state_dict()[name].data.copy_(models_H[h][name])
                    cluster_models_lambda[h][n_en][name].data.copy_(models_H_lambda[h][name])
                cluster_weights[h][n_en] = weights_H[h]
            # update the servers
            # servers_list[n_en] = server
            # update the nodes model
            # nodes_list[h][n_en] = nodes
        # # ensemble
        # cluster_models[n_en]=server.model

        # test ensemble
        print('test ensemble\n')
        combine = False
        if not combine:
            cluster_models_lst = [item for sublist in cluster_models for item in sublist]
        # combine hypothesis
        else:
            cluster_models_comb = [model() for i in range(n_ensemble)]
            for k in range(n_ensemble):
                for h in range(H):
                    for name, param in cluster_models[h][k].named_parameters():
                        if h == 0:
                            cluster_models_comb[k].state_dict()[name].data.copy_(cluster_models[h][k].state_dict()[name])
                        else:
                            cluster_models_comb[k].state_dict()[name] = cluster_weights[h][n_en]*cluster_models_comb[k].state_dict()[name] + cluster_weights[h][n_en]*cluster_models[h][k].state_dict()[name]
            cluster_models_lst = cluster_models_comb
        for j in range(num_nodes):
            nodes[j].local_ensemble_test(cluster_models_lst, voting = 'hard') #cluster_models_comb
        server.acc(nodes, weight_list)

    # log
    log(os.path.basename(__file__)[:-3] + add_(n_ensemble)+add_(H) + add_(split_para), nodes, server)

    return cluster_models