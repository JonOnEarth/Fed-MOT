# deprecated, see fed-mot-GNN.py
# from fedbase.utils.data_loader import data_process, log
# from fedbase.nodes.node_fl_mot import node
# from fedbase.utils.tools import add_
# from fedbase.server.server_fl_mot import server_class
# import torch
# from torch.utils.data import DataLoader
# import torch.optim as optim
# from fedbase.model.model import CNNCifar, CNNMnist
# import os
# import sys
# import inspect
# from functools import partial
# import copy
# import torch.nn as nn

# def run(dataset_splited, batch_size, K, num_nodes, model, objective, optimizer, global_rounds, local_steps, reg = None, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),\
#          finetune=False, finetune_steps = None, weight_method = 'data_size',aggregated_method='GA',assign_method='clustering'):
#     # dt = data_process(dataset)
#     # train_splited, test_splited = dt.split_dataset(num_nodes, split['split_para'], split['split_method'])
#     train_splited, test_splited, split_para = dataset_splited
    
#     server = server_class(device)
#     server.assign_model(model())
#     model_lambda = dict()
#     for name, param in server.model.named_parameters():
#         model_lambda[name] = torch.ones_like(param)
#     server.assign_model_lambda(model_lambda)

#     nodes = [node(i, device) for i in range(num_nodes)]
#     # local_models = [model() for i in range(num_nodes)]
#     # local_loss = [objective() for i in range(num_nodes)]

#     for i in range(num_nodes):
#         # data
#         # print(len(train_splited[i]), len(test_splited[i]))
#         nodes[i].assign_train(DataLoader(train_splited[i], batch_size=batch_size, shuffle=True))
#         nodes[i].assign_test(DataLoader(test_splited[i], batch_size=batch_size, shuffle=False))
#         # model
#         nodes[i].assign_model(model())
#         nodes[i].assign_model_lambda(model_lambda)
#         # objective
#         nodes[i].assign_objective(objective())
#         # optim
#         nodes[i].assign_optim(optimizer(model().parameters()))

#     del train_splited, test_splited

#     # initialize K cluster model
#     cluster_models = [model() for i in range(K)]
#     cluster_models_lambda = [model_lambda for i in range(K)]

#     # train!
#     weight_list = [nodes[i].data_size/sum([nodes[i].data_size for i in range(num_nodes)]) for i in range(num_nodes)]
#     for t in range(global_rounds):
#         print('-------------------Global round %d start-------------------' % (t))
#         # assign client to cluster
#         assignment = [[] for _ in range(K)]
#         if assign_method == 'loss':
#             for i in range(num_nodes):
#                 m = 0
#                 for k in range(1, K):
#                     # print(nodes[i].local_train_loss(cluster_models[m]), nodes[i].local_train_loss(cluster_models[k]))
#                     if nodes[i].local_train_loss(cluster_models[m])>=nodes[i].local_train_loss(cluster_models[k]):
#                         m = k
#                 assignment[m].append(i)
#                 nodes[i].label = m
#                 nodes[i].assign_model(cluster_models[m])
#                 nodes[i].assign_model_lambda(cluster_models_lambda[m])
#                 nodes[i].assign_optim(optimizer(nodes[i].model.parameters()))
#             # print(server.clustering)
#             server.clustering['label'].append(assignment)
#             print(assignment)
#             print([len(assignment[i]) for i in range(len(assignment))])

#             # local update
#             for j in range(num_nodes):
#                 # if not reg:
#                 #     nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step))
#                 # else:
#                 #     nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_fedprox, reg_model = server.aggregate(nodes, list(range(num_nodes))), lam= reg))
#                 nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_bayes, reg_model = cluster_models[nodes[j].label], reg_model_lambda = cluster_models_lambda[nodes[j].label]))
#         elif assign_method == 'clustering':
#             # local update
#             for j in range(num_nodes):
#                 if t == 0:
#                     # random choose from 0 to K
#                     nodes[j].label = torch.randint(0, K, (1,)).item()
#                 nodes[j].assign_model(cluster_models[nodes[j].label])
#                 nodes[j].assign_model_lambda(cluster_models_lambda[nodes[j].label])
#                 nodes[j].assign_optim(optimizer(nodes[j].model.parameters()))
#                 nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_bayes, reg_model = cluster_models[nodes[j].label], reg_model_lambda = cluster_models_lambda[nodes[j].label]))
#             server.weighted_clustering(nodes, list(range(num_nodes)), K)
#             for j in range(K):
#                 assignment[j] = [i for i in list(range(num_nodes)) if nodes[i].label == j]
#         # server aggregation and distribution by cluster
#         for k in range(K):
#             if len(assignment[k])>0:
#                 if weight_method == 'uniform':
#                     weight_ls = [1/len(assignment[k]) for i in assignment[k]]
#                 elif weight_method == 'data_size':
#                     weight_ls = [nodes[i].data_size/sum([nodes[i].data_size for i in assignment[k]]) for i in assignment[k]]
#                 elif weight_method == 'loss':
#                     node_i_loss = [nodes[i].get_ce_loss() for i in assignment[k]]
#                     weight_ls = [nodes[i].weight/sum(nodes[i].weight for i in assignment[k]) for i in assignment[k]]

#                 model_k, model_k_lambda = server.aggregate_bayes([nodes[i].model for i in assignment[k]], [nodes[i].model_lambda for i in assignment[k]], weight_ls,aggregated_method)
#                 server.distribute([nodes[i].model for i in assignment[k]], model_k)
#                 server.distribute_lambda([nodes[i].model_lambda for i in assignment[k]], model_k_lambda)
#                 for name, param in cluster_models[k].named_parameters():
#                     cluster_models[k].state_dict()[name].data.copy_(model_k[name])
#                 # cluster_models[k].load_state_dict(model_k)
#                 cluster_models_lambda[k] = model_k_lambda

#         # test accuracy
#         for i in range(num_nodes):
#             nodes[i].local_test()
#         server.acc(nodes, weight_list)

#     # save the log
#     log(os.path.basename(__file__)[:-3] + add_(K)  + add_(split_para), nodes, server)

#         # if not finetune:
#         #     assign = [[i for i in range(num_nodes) if nodes[i].label == k] for k in range(K)]
#         #     # log
#         #     log(os.path.basename(__file__)[:-3] + add_(K) + add_(reg) + add_(split_para), nodes, server)
#         #     return cluster_models, assign
#         # else:
#         #     if not finetune_steps:
#         #         finetune_steps = local_steps
#         #     # fine tune
#         #     for j in range(num_nodes):
#         #         if not reg:
#         #             nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step))
#         #         else:
#         #             nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_fedprox, reg_model = server.aggregate(nodes, list(range(num_nodes))), lam= reg))
#         #         nodes[j].local_test()
#         #     server.acc(nodes, weight_list)
#         #     # log
#         #     log(os.path.basename(__file__)[:-3] + add_('finetune') + add_(K) + add_(reg) + add_(split_para), nodes, server)
#         #     return [nodes[i].model for i in range(num_nodes)]