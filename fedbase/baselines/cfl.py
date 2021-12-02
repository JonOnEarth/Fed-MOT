from fedbase.utils.data_loader import data_process
from fedbase.nodes.node import node
from fedbase.server.server import server_class
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from fedbase.model.model import CNNCifar, CNNMnist


def cfl(dir, dataset, batch_size, K, num_nodes, model, objective, optimizer, global_rounds, local_epochs, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    dt = data_process(dir, dataset)
    train_splited,test_splited = dt.split_dataset(num_nodes, 2, method='class')

    server = server_class()
    server.assign_model(model(), device)

    nodes = [node(i) for i in range(num_nodes)]
    local_models = [model() for i in range(num_nodes)]
    local_loss = [objective() for i in range(num_nodes)]

    for i in range(num_nodes):
        # data
        # print(len(train_splited[i]), len(test_splited[i]))
        nodes[i].assign_train(DataLoader(train_splited[i], batch_size=batch_size, shuffle=True))
        nodes[i].assign_test(DataLoader(test_splited[i],batch_size=batch_size, shuffle=False))
        # model
        nodes[i].assign_model(local_models[i], device)
        # objective
        nodes[i].assign_objective(local_loss[i])
        # optim
        # nodes[i].assign_optim(optim.Adam(nodes[i].model.parameters()))
        nodes[i].assign_optim(optim.SGD(nodes[i].model.parameters(), lr=0.001, momentum=0.9))

    # initialize parameters to nodes
    server.distribute(nodes, list(range(num_nodes)))

    # train!
    for i in range(global_rounds):
        print('-------------------Global round %d start-------------------' % (i))
        # single-processing!
        for j in range(num_nodes):
            nodes[j].local_update(local_epochs, device)
            nodes[j].local_test(device)
        # server clustering
        server.weighted_clustering(nodes, list(range(num_nodes)), K)
        
        # server aggregation and distribution
        for i in range(K):
            # print( [j for j in list(range(num_nodes)) if nodes[j].label==i])
            server.aggregate(nodes, [j for j in list(range(num_nodes)) if nodes[j].label==i], device)
            server.distribute(nodes, [j for j in list(range(num_nodes)) if nodes[j].label==i])
        server.acc(nodes, list(range(num_nodes)))