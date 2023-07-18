# Single MHT
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

def run(dataset_splited, batch_size, num_nodes, model, objective, optimizer, global_rounds, local_steps,  H, device,\
        node_weight_type='data_size', cluster_method='kmeans', personalized_weight_type='loss', fusion_local='prune'): #weight_type='data_size'
#   
 # train_splited, test_splited = dt.split_dataset(num_nodes, split['split_para'], split['split_method'])
    train_splited, test_splited, split_para = dataset_splited
    server = server_class(device)
    server.assign_model(model())
    model_lambda = dict()
    for name, param in server.model.named_parameters():
        model_lambda[name] = torch.ones_like(param)
    server.assign_model_lambda(model_lambda)
    
    cluster_models = [model().to(device) for h in range(H)]
    cluster_models_lambda = [model_lambda  for h in range(H)]
    cluster_weights = [1/H for h in range(H)]
    
    # nodes_list = []
    # for h in range(H):
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
        # nodes_list.append(nodes)
    del train_splited, test_splited
    weight_list = [nodes[i].data_size/sum([nodes[i].data_size for i in range(num_nodes)]) for i in range(num_nodes)]
    
    for round in range(global_rounds):
        print('-------------------Global round %d start-------------------' % (round))
        # single-processing!
        nodes_weight = []
        nodes_H_M = []
        for j in range(num_nodes):
            rs_new = []
            node_H_j = []
            for h in range(H):
                node_h = copy.deepcopy(nodes[j])
                node_h.assign_model(cluster_models[h])
                node_h.assign_model_lambda(cluster_models_lambda[h])
                if node_weight_type == 'loss':
                    ce_loss = node_h.get_ce_loss()
                    node_h_weight = node_h.weight
                elif node_weight_type == 'data_size':
                    node_h_weight = node_h.data_size/sum([nodes[i].data_size for i in range(num_nodes)])
                else: # node_weight_type = 'uniform'
                    node_h_weight = 1/num_nodes
                r_new = node_h_weight * cluster_weights[h]
                node_h.local_update_steps(local_steps, partial(node_h.train_single_step_bayes, reg_model = cluster_models[h], reg_model_lambda = cluster_models_lambda[h]))
                node_H_j.append(node_h)
                rs_new.append(r_new)
            rs_new = torch.tensor(rs_new)
            # rs_new = torch.tensor(rs_new)/sum(torch.tensor(rs_new))
            # prune the models locally before sending to the server
            if fusion_local == 'prune':
                idex_list = server.prune_local(rs_new, 1)
                node_H_j = [node_H_j[i] for i in idex_list]
                rs_new = rs_new[idex_list]
            elif fusion_local == 'merge':
                pass

            nodes_H_M.extend(node_H_j)
            nodes_weight.extend(rs_new)
            
        server.weighted_clustering(nodes_H_M, list(range(len(nodes_H_M))), H)#, weight_type='loss')

        # aggregate the models
        
        for h in range(H):
            assign_ls = [i for i in list(range(len(nodes_H_M))) if nodes_H_M[i].label==h]
            # if weight_type == 'data_size':
            #     weight_ls = [nodes_H_M[i].data_size/sum([nodes_H_M[i].data_size for i in assign_ls]) for i in assign_ls]
            # else:
            weight_ls = [nodes_weight[i] for i in assign_ls]
            weight_ls = torch.tensor(weight_ls)/sum(torch.tensor(weight_ls))
            model_h, model_h_lambda, weight_h = server.aggregate_bayes([nodes_H_M[i].model for i in range(num_nodes)],\
                [nodes_H_M[i].model_lambda for i in range(num_nodes)], weight_ls, aggregated_method='AA')
            
            for name, param in cluster_models[h].named_parameters():
                cluster_models[h].state_dict()[name].data.copy_(model_h[name])
                cluster_models_lambda[h][name].data.copy_(model_h_lambda[name])
            
            cluster_weights[h] = weight_h
        # normalize the weights
        cluster_weights = torch.tensor(cluster_weights)/sum(torch.tensor(cluster_weights))
        
        # print the ensemble accuracy
        print('test ensemble\n')
        for j in range(num_nodes):
            nodes[j].local_ensemble_test(cluster_models, voting = 'soft')
        server.acc(nodes, weight_list)

    # log
    log(os.path.basename(__file__)[:-3] + 'H'+add_(H) + add_(split_para), nodes, server)
