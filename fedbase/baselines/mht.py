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
from fedbase.utils import assignment_func

def run(dataset_splited, batch_size, K, num_nodes, model, objective, optimizer, global_rounds, local_steps, \
    reg_lam = None, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), finetune=False, finetune_steps = None,\
         bayes=True,num_assign=2,hypothesis=2):
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

    cluster_models_assignments_hypothesis = [copy.deepcopy(cluster_models) for i in range(hypothesis)]
    cluster_models_lambda_assignments_hypothesis = [copy.deepcopy(cluster_models_lambda) for i in range(hypothesis)]
    cost_ks_assignments_hypothesis = [torch.log(torch.tensor(1/hypothesis))] * hypothesis

    # server_accuracy = torch.zeros(global_rounds)
    # train!
    for t in range(global_rounds):
        print('-------------------Global round %d start-------------------' % (t))

        cost_ks_assignments_hypothesis_new = []
        cluster_models_assignments_hypothesis_new = []
        cluster_models_lambda_assignments_hypothesis_new = []

        for h in range(hypothesis):
            
            cost_matrix = torch.zeros((num_nodes, K))
            cluster_models = cluster_models_assignments_hypothesis[h]
            cluster_models_lambda = cluster_models_lambda_assignments_hypothesis[h]
            for j in range(num_nodes):
                for k in range(0, K):
                    # build the cost matrix
                    loss = nodes[j].local_train_loss(cluster_models[k])
                    cost_matrix[j][k] = loss
            assignments = assignment_func.get_num_assignments(cost_matrix.numpy(), num_assign)
            print('assignments: ', assignments)
            trained_index = []
            trained_nodes = []
            nodes_assignments = [copy.deepcopy(nodes) for a in range(len(assignments))]
            cost_ks_assignments = []
            cluster_models_assignments = []
            cluster_models_lambda_assignments = []
            for a, assign in enumerate(assignments): # assign is a list of cluster index of nodes
                nodes = nodes_assignments[a]
                for j in range(num_nodes):
                    nodes[j].label = assign[j] # = k
                    nodes[j].assign_model(cluster_models[assign[j]])
                    nodes[j].assign_model_lambda(cluster_models_lambda[assign[j]])
                    # check if the node is trained in this round
                    if [j, assign[j]] in trained_index:
                        # return the index of [j, assign[j]] in trained_index
                        index = trained_index.index([j, assign[j]])
                        nodes[j] = copy.deepcopy(trained_nodes[index])
                    else:
                        # local update
                        if t == 0 or not bayes:
                            nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step))
                        else:
                            nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_bayes, reg_model = cluster_models[nodes[j].label], reg_model_lambda = cluster_models_lambda[nodes[j].label]))
                        trained_index.append([j, assign[j]])
                        trained_nodes.append(nodes[j])
                
                # calculate the cost of the assignment for each cluster
                cost_ks = []
                cluster_models_temp = copy.deepcopy(cluster_models_assignments_hypothesis[h])
                cluster_models_lambda_temp = copy.deepcopy(cluster_models_lambda_assignments_hypothesis[h])
                for k in range(K):
                    cost_k = sum([cost_matrix[j][k] for j in range(num_nodes) if nodes[j].label == k])
                    if sum([nodes[j].label == k for j in range(num_nodes)]) == 0:
                        cost_k = 0
                    cost_ks.append(cost_k)
                    # for this assignment, aggregate the assigned nodes' models
                    assign_ls = [i for i in list(range(num_nodes)) if nodes[i].label==k]
                    if assign_ls == []:
                        continue
                    weight_ls = [nodes[i].data_size/sum([nodes[i].data_size for i in assign_ls]) for i in assign_ls]
                    weight_ls = torch.tensor(weight_ls)
                    if not bayes:
                        model_k = server.aggregate([nodes[i].model for i in assign_ls], weight_ls)
                        server.distribute([nodes[i].model for i in assign_ls], model_k)
                        cluster_models_temp[k].load_state_dict(model_k)
                    else:
                        model_k, model_lambda_k,_ = server.aggregate_bayes([nodes[i].model for i in assign_ls], [nodes[i].model_lambda for i in assign_ls], weight_ls)
                        server.distribute([nodes[i].model for i in assign_ls], model_k)
                        server.distribute_lambda([nodes[i].model_lambda for i in assign_ls], model_lambda_k)
                    
                        for name, param in cluster_models_temp[k].named_parameters():
                            cluster_models_temp[k].state_dict()[name].data.copy_(model_k[name])
                        cluster_models_lambda_temp[k] = model_lambda_k
                cost_ks_assignments.append(sum(cost_ks)+cost_ks_assignments_hypothesis[h]) # here sum of all cost of assignments + previous cost of this hypothesis
                cluster_models_assignments.append(cluster_models_temp)
                cluster_models_lambda_assignments.append(cluster_models_lambda_temp)

                # # test accuracy
                # if accuracy_type == 'single':
                print('test single of assignment %d\n' % (a))
                if a == 0 and h == 0:
                    type='save'
                    # only save the best assignment metrics
                    for j in range(num_nodes):
                        nodes[j].local_test()
                    global_test_metrics = server.acc2(nodes, weight_list, type)
                    server.clustering['label'].append(assign)
                else:
                    for j in range(num_nodes):
                        nodes[j].local_test()
                    server.acc2(nodes, weight_list)

                # elif accuracy_type == 'ensemble':
                #     print('test ensemble\n')
                #     for j in range(num_nodes):
                #         nodes[j].local_ensemble_test(cluster_models, voting = 'soft')
                #     server.acc(nodes, weight_list)

            cost_ks_assignments_hypothesis_new.extend(cost_ks_assignments)
            cluster_models_assignments_hypothesis_new.extend(cluster_models_assignments)
            cluster_models_lambda_assignments_hypothesis_new.extend(cluster_models_lambda_assignments)

        # prune the temp hypothesis to hypothesis
        cost_ks_assignments_hypothesis_new = torch.tensor(cost_ks_assignments_hypothesis_new)
        print('cost_ks_assignments_hypothesis_new: ', cost_ks_assignments_hypothesis_new)
        sorted_index = torch.argsort(cost_ks_assignments_hypothesis_new)
        cap = sorted_index[:hypothesis]
        cost_ks_assignments_hypothesis = torch.tensor([cost_ks_assignments_hypothesis_new[i] for i in cap])
        print('cost_ks_assignments_hypothesis: ', cost_ks_assignments_hypothesis)
        cluster_models_assignments_hypothesis = [cluster_models_assignments_hypothesis_new[i] for i in cap]
        cluster_models_lambda_assignments_hypothesis = [cluster_models_lambda_assignments_hypothesis_new[i] for i in cap]
        # normalize the cost
        if hypothesis > 1:
            bs = torch.zeros(len(cost_ks_assignments_hypothesis))
            for i,los in enumerate(cost_ks_assignments_hypothesis):
                bs[i] = los-cost_ks_assignments_hypothesis.max()
            cost_ks_assignments_hypothesis = bs - torch.log(1+sum(torch.exp(bs[:hypothesis-1])))

    
        # test accuracy
        # if accuracy_type == 'single':
        #     for j in range(num_nodes):
        #         nodes[j].local_test()
        #     server.acc(nodes, weight_list)
        # elif accuracy_type == 'ensemble':
        #     print('test ensemble\n')
        print('test ensemble of each round \n')
        for j in range(num_nodes):
            nodes[j].local_ensemble_test(cluster_models_assignments_hypothesis[0], voting = 'max')
        server.acc(nodes, weight_list)


    if not finetune:
        assign = [[i for i in range(num_nodes) if nodes[i].label == k] for k in range(K)]
        # log
        log(os.path.basename(__file__)[:-3] +add_(K) + add_(num_assign)+ add_(hypothesis) + add_(split_para), nodes, server)
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
        log(os.path.basename(__file__)[:-3] + add_('finetune') + add_(K) + add_(reg_lam) + add_(split_para), nodes, server)
        return [nodes[i].model for i in range(num_nodes)]