from fedbase.utils.data_loader import data_process, log
from fedbase.utils.tools import add_
# from fedbase.nodes.node import node
# from fedbase.server.server import server_class
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from functools import partial
from fedbase.server.server_fl_mot import server_class
from fedbase.nodes.node_fl_mot import node

def run(dataset_splited, batch_size, num_nodes, model, objective, optimizer, global_rounds, local_steps,\
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), log_file=True, finetune=False, finetune_steps = None,\
          weight_method = 'loss',aggregated_method='AA'):
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
    weight_list = [nodes[i].data_size/sum([nodes[i].data_size for i in range(num_nodes)]) for i in range(num_nodes)]
    
    # train!
    for t in range(global_rounds):
        print('-------------------Global round %d start-------------------' % (t))
        # single-processing!
        for j in range(num_nodes):
            nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step_bayes,reg_model = server.model, reg_model_lambda=server.model_lambda))
        if weight_method == 'uniform':
            weights = [1/num_nodes for i in range(num_nodes)]
        elif weight_method == 'data_size':
            weights = weight_list
        elif weight_method == 'loss':
            for j in range(num_nodes):
                nodes[j].get_ce_loss()
            weights = [nodes[i].weight/sum(nodes[i].weight for i in range(num_nodes)) for i in range(num_nodes)]
        # server aggregation and distribution
        new_param, new_lambda = server.aggregate_bayes([nodes[i].model for i in range(num_nodes)], [nodes[i].model_lambda for i in range(num_nodes)], weights, aggregated_method)
        # server.model.load_state_dict()
        for name, param in server.model.named_parameters():
            server.model.state_dict()[name].data.copy_(new_param[name])
            server.model_lambda[name].data.copy_(new_lambda[name])
        server.distribute([nodes[i].model for i in range(num_nodes)],new_param)
        server.distribute_lambda([nodes[i].model_lambda for i in range(num_nodes)],new_lambda)
        # test accuracy
        for j in range(num_nodes):
            nodes[j].local_test()
        server.acc(nodes, weight_list)

    if not finetune:
        # log
        if log_file:
            log(os.path.basename(__file__)[:-3] + add_(split_para), nodes, server)
        return server.model
    else:
        if not finetune_steps:
            finetune_steps = local_steps
        # fine tune
        for j in range(num_nodes):
            nodes[j].local_update_steps(finetune_steps, partial(nodes[j].train_single_step))
            nodes[j].local_test()
        server.acc(nodes, weight_list)
        # log
        log(os.path.basename(__file__)[:-3] + add_('finetune') + add_(split_para), nodes, server)
        return [nodes[i].model for i in range(num_nodes)]