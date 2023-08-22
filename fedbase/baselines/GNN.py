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
    reg_lam = None, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), finetune=False, finetune_steps = None,\
        assign_method='ifca', bayes=True, warm_up=False, warm_up_rounds=2,accuracy_type='single'):
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

    # initialize K cluster model
    cluster_models = [model() for i in range(K)]
    cluster_models_lambda = [model_lambda for i in range(K)]

    assign_method_copy = copy.deepcopy(assign_method)
    # train!
    for t in range(global_rounds):
        print('-------------------Global round %d start-------------------' % (t))
        if assign_method== 'ifca' and warm_up==True and t <= warm_up_rounds:
            assign_method = 'wecfl'
        else:
            assign_method = assign_method_copy
        
        if assign_method == 'ifca':
            # local update
            assignment = [[] for i in range(K)]
            for j in range(num_nodes):
                m = 0
                for k in range(1, K):
                    # print(nodes[i].local_train_loss(cluster_models[m]), nodes[i].local_train_loss(cluster_models[k]))
                    if nodes[j].local_train_loss(cluster_models[m])>=nodes[j].local_train_loss(cluster_models[k]):
                        m = k
                assignment[m].append(j)
                nodes[j].label = m
                nodes[j].assign_model(cluster_models[m])
                nodes[j].assign_model_lambda(cluster_models_lambda[m])
                # nodes[j].assign_optim(optimizer(nodes[j].model.parameters()))
                # local update
                if t == 0 or not bayes:
                    nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step))
                else:
                    nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_bayes, reg_model = cluster_models[nodes[j].label], reg_model_lambda = cluster_models_lambda[nodes[j].label]))
            # print(server.clustering)
            server.clustering['label'].append(assignment)
            # print('clustering: ', assignment)
            print([nodes[j].label for j in range(num_nodes)])
        elif assign_method == 'wecfl':
            for j in range(num_nodes):
                if t == 0 or not bayes:
                    nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step)) 
                else:
                    nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_bayes, reg_model = cluster_models[nodes[j].label], reg_model_lambda = cluster_models_lambda[nodes[j].label]))
            # server clustering, assign_method=='wecfl'
            server.weighted_clustering(nodes, list(range(num_nodes)), K)

    
        # server aggregation and distribution by cluster
        for k in range(K):
            assign_ls = [i for i in list(range(num_nodes)) if nodes[i].label==k]
            if assign_ls == []:
                continue
            weight_ls = [nodes[i].data_size/sum([nodes[i].data_size for i in assign_ls]) for i in assign_ls]
            weight_ls = torch.tensor(weight_ls)
            # model_k = server.aggregate([nodes[i].model for i in assign_ls], weight_ls)
            if not bayes:
                model_k = server.aggregate([nodes[i].model for i in assign_ls], weight_ls)
                server.distribute([nodes[i].model for i in assign_ls], model_k)
                cluster_models[k].load_state_dict(model_k)
            else:
                model_k, model_lambda_k,_ = server.aggregate_bayes([nodes[i].model for i in assign_ls], [nodes[i].model_lambda for i in assign_ls], weight_ls)
                server.distribute([nodes[i].model for i in assign_ls], model_k)
                server.distribute_lambda([nodes[i].model_lambda for i in assign_ls], model_lambda_k)
            
                for name, param in cluster_models[k].named_parameters():
                    cluster_models[k].state_dict()[name].data.copy_(model_k[name])
                cluster_models_lambda[k] = model_lambda_k

        # test accuracy
        if accuracy_type == 'single':
            for j in range(num_nodes):
                nodes[j].local_test()
            server.acc(nodes, weight_list)
        elif accuracy_type == 'ensemble':
            print('test ensemble\n')
            for j in range(num_nodes):
                nodes[j].local_ensemble_test(cluster_models, voting = 'soft')
            server.acc(nodes, weight_list)
    
    if not finetune:
        assign = [[i for i in range(num_nodes) if nodes[i].label == k] for k in range(K)]
        # log
        log(os.path.basename(__file__)[:-3] + add_(assign_method)+add_(K) + add_(reg_lam) + add_(warm_up), nodes, server)
        return cluster_models, assign
    else:
        if not finetune_steps:
            finetune_steps = local_steps
        # fine tune
        for j in range(num_nodes):
            if not reg_lam:
                nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step))
            else:
                nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_fedprox, reg_model = cluster_models[nodes[j].label], reg_lam= reg_lam))
            nodes[j].local_test()
        server.acc(nodes, weight_list)
        # log
        log(os.path.basename(__file__)[:-3] + add_('finetune')+add_(assign_method) + add_(K) + add_(reg_lam) + add_(split_para), nodes, server)
        return [nodes[i].model for i in range(num_nodes)]