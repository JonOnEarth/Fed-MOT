from fedbase.utils.data_loader import data_process, log
from fedbase.utils.tools import add_
# from fedbase.nodes.node import node
from fedbase.nodes.node_fl_mot import node
# from fedbase.server.server import server_class
from fedbase.server.server_fl_mot import server_class
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from functools import partial

def run(dataset_splited, batch_size, num_nodes, model, objective, optimizer, global_rounds, local_steps,\
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), log_file=True,\
          finetune=False, finetune_steps = None, path='log/'):
    # dt = data_process(dataset)
    # train_splited, test_splited = dt.split_dataset(num_nodes, split['split_para'], split['split_method'])
    train_splited, test_splited, split_para = dataset_splited

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

    del train_splited, test_splited

    # initialize parameters to nodes
    server.distribute([nodes[i].model for i in range(num_nodes)])
    weight_list = [nodes[i].data_size/sum([nodes[i].data_size for i in range(num_nodes)]) for i in range(num_nodes)]
    
    # train!
    for t in range(global_rounds):
        print('-------------------Global round %d start-------------------' % (t))
        # single-processing!
        for j in range(num_nodes):
            # nodes[j].local_update_steps(local_steps, partial(nodes[j].train_single_step))
            nodes[j].local_update_epochs(local_steps, partial(nodes[j].train_single_step))
            # nodes[j].local_test_conf()
            print('node %d local update done' % (j))
        # server aggregation and distribution
        # server.model.load_state_dict(server.aggregate([nodes[i].model for i in range(num_nodes)], weight_list))
        # weight_list = [1/num_nodes for i in range(num_nodes)]
        model_k = server.aggregate([nodes[i].model for i in range(num_nodes)], weight_list)
        server.model.load_state_dict(model_k)
        server.distribute([nodes[i].model for i in range(num_nodes)])
        # test accuracy
        for j in range(num_nodes):
        #     nodes[j].local_test()
        # server.acc(nodes, weight_list)
            nodes[j].local_test_conf()
        server.acc_conf(nodes, weight_list)

    if not finetune:
        # log
        if log_file:
            log(os.path.basename(__file__)[:-3] + add_(split_para), nodes, server, path=path)
        return server.model
    else:
        if not finetune_steps:
            finetune_steps = local_steps
        # fine tune
        for j in range(num_nodes):
            nodes[j].local_update_steps(finetune_steps, partial(nodes[j].train_single_step))
        #     nodes[j].local_test()
        # server.acc(nodes, weight_list)
            nodes[j].local_test_conf()
        server.acc_conf(nodes, weight_list)
        # log
        log(os.path.basename(__file__)[:-3] + add_('finetune') + add_(split_para), nodes, server)
        return [nodes[i].model for i in range(num_nodes)]