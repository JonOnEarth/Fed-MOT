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

def run(dataset_splited, batch_size, K, num_nodes, model, objective, optimizer, global_rounds, local_steps, \
    reduction = None, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), finetune=False, finetune_steps = None, temperature=torch.tensor(1.0)):
    # dt = data_process(dataset)
    # train_splited, test_splited = dt.split_dataset(num_nodes, split['split_para'], split['split_method'])
    train_splited, test_splited, split_para = dataset_splited
    server = server_class(device)
    server.assign_model(model())
    # define the parameter lambda
    model_lambda = dict()
    for name, param in server.model.named_parameters():
        model_lambda[name] = torch.ones_like(param)
    server.assign_model_lambda(model_lambda)

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
    
    del train_splited, test_splited

    # initialize parameters to nodes
    server.distribute([nodes[i].model for i in range(num_nodes)])
    # weight_list = [nodes[i].data_size/sum([nodes[i].data_size for i in range(num_nodes)]) for i in range(num_nodes)]
    weight_list = [1.0 for i in range(num_nodes)]

    # initialize K cluster model
    # this method can be defined outside your model class
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=1.0)
            # torch.nn.init.zero_(m.bias)
            
    # define init method inside your model class
    
    cluster_models = [model().apply(weights_init) for i in range(K)]
    cluster_models_lambda = [model_lambda for i in range(K)]

    
    # train!
    for t in range(global_rounds):
        print('-------------------Global round %d start-------------------' % (t))
        
        # local update for weight and JPDA training
        nodes_k_m = [[] for j in range(num_nodes)]
        nodes_k_m_weight = [[1] for j in range(num_nodes)]
        for j in range(num_nodes):
            nodes_k = [[] for i in range(K)]
            nodes_k_weight = [[] for i in range(K)]
            for i in range(K):
                nodes[j].assign_model(cluster_models[i])
                nodes[j].assign_model_lambda(cluster_models_lambda[i])
                nodes[j].get_ce_loss(temperature)
                if reduction == 'JPDA':
                    nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_bayes))
                elif reduction == 'GNN':
                    pass
                nodes_k[i].append(nodes[j])
                nodes_k_weight[i].append(nodes[j].weight)
            nodes_k_m[j] = nodes_k
            nodes_k_m_weight[j] = nodes_k_weight
        # server aggregation and distribution by cluster
        if reduction == 'JPDA':
            for j in range(K):
                model_k, model_k_lambda = server.aggregate_bayes([nodes_k_m[i][j].model for i in range(num_nodes)],\
                    [nodes_k_m[i][j].model_lambda for i in range(num_nodes)],\
                          [nodes_k_m_weight[i][j] for i in range(num_nodes)],aggregated_method='AA')
                cluster_models[j].load_state_dict(model_k)
                cluster_models_lambda[j] = model_k_lambda
            
            # test
            for j in range(num_nodes):
                accs = []
                weights = []
                for i in range(K):
                    nodes[j].assign_model(cluster_models[i])
                    nodes[j].assign_model_lambda(cluster_models_lambda[i])
                    nodes[j].get_ce_loss(temperature)
                    nodes[j].test()
                    accs.append(nodes[j].test_acc)
                    weights.append(nodes[j].weight)
                accs_node = sum(torch.tensor(accs)*torch.tensor(weights)/sum(weights))
                print('Node %d test acc: %f' % (j, accs_node))


        elif reduction == 'GNN':
            # get the biggest value of nodes_k_m_weight of each node and its index
            nodes_k_m_weight_max = [max(nodes_k_m_weight[j]) for j in range(num_nodes)]
            nodes_k_m_weight_max_index = [nodes_k_m_weight[j].index(max(nodes_k_m_weight[j])) for j in range(num_nodes)]
            # reduce the nodes_k_m_weight_max to 1 dimension and normalize it
            nodes_k_m_weight_max = torch.tensor(nodes_k_m_weight_max) / sum(torch.tensor(nodes_k_m_weight_max))
            # turn two dimension to one dimension
            nodes_k_m_weight_max = torch.tensor(nodes_k_m_weight_max).view(-1)
            print('nodes_k_m_weight_max_index:', nodes_k_m_weight_max_index)
            # train
            for j in range(num_nodes):
                nodes[j].assign_model(nodes_k_m[j][nodes_k_m_weight_max_index[j]][0].model)
                nodes[j].assign_model_lambda(nodes_k_m[j][nodes_k_m_weight_max_index[j]][0].model_lambda)
                nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_bayes))
            # aggregate
            for i in range(K):
                assign_ls = [j for j in list(range(num_nodes)) if nodes_k_m_weight_max_index[j]==i]
                weight_ls = [nodes_k_m_weight_max[j] for j in list(range(num_nodes)) if nodes_k_m_weight_max_index[j]==i]
                model_k, model_k_lambda = server.aggregate_bayes([nodes[j].model for j in assign_ls], [nodes[j].model_lambda for j in assign_ls], weight_ls, aggregated_method='GA')
                server.distribute([nodes[j].model for j in assign_ls], model_k)
                server.distribute_lambda([nodes[j].model_lambda for j in assign_ls], model_k_lambda)
                for name, param in cluster_models[i].named_parameters():
                    cluster_models[i].state_dict()[name].data.copy_(model_k[name])
                # cluster_models[i].load_state_dict(model_k)
                # for j in assign_ls:
                #     nodes[j].assign_model(cluster_models[i]) 
                #     nodes[j].assign_model_lambda(model_k_lambda)
                
                # cluster_models[i] = model_k
                cluster_models_lambda[i] = model_k_lambda
            # test accuracy
            for j in range(num_nodes):
                nodes[j].local_test()
            server.acc(nodes, nodes_k_m_weight_max)

    # if not finetune:
    #     assign = [[i for i in range(num_nodes) if nodes[i].label == k] for k in range(K)]
    #     # log
    #     log(os.path.basename(__file__)[:-3] + add_(K)  + add_(split_para), nodes, server)
    #     return cluster_models, assign
    # else:
    #     if not finetune_steps:
    #         finetune_steps = local_steps
    #     # fine tune
    #     for j in range(num_nodes):
    #         # if not reg_lam:
    #         #     nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step))
    #         # else:
    #         #     nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_fedprox, reg_model = cluster_models[nodes[j].label], reg_lam= reg_lam))
    #         nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_bayes))
    #         nodes[j].local_test()
    #     server.acc(nodes, weight_list)
    #     # log
    #     log(os.path.basename(__file__)[:-3] + add_('finetune') + add_(K)  + add_(split_para), nodes, server)
    #     return [nodes[i].model for i in range(num_nodes)]