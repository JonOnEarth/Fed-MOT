from fedbase.utils.data_loader import data_process
from fedbase.nodes.node import node
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

def central(dir, dataset, batch_size, nodes, model, objective, optimizer, global_rounds, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    dt = data_process(dir, dataset)
    trainset,testset = dt.train_dataset, dt.test_dataset
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

    nodes0 = node(0)
    nodes0.assign_train(trainloader)
    nodes0.assign_test(testloader)
    nodes0.assign_model(model, device)
    nodes0.assign_objective(objective)
    # optimizer = optim.SGD(nodes0.model.parameters(), lr=0.01, momentum=0.9)
    nodes0.assign_optim(optimizer)

    print('-------------------start-------------------')
    for i in range(global_rounds):
        nodes0.local_update(1, device)
        nodes0.local_test(device)

    # log
    log(os.path.basename(__file__)[:-3], nodes, server)