# fedavg algorithm for jammer dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy
import numpy as np
import time
import os
import sys
import argparse
import logging
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch.optim as optim

# from utils import load_dataloader
from fedbase.utils import loader_gnss
from fedbase.model.model import *

class server(object):
    def __init__(self, model, device):
       
        self.device = device
        self.test_metrics = []
        self.clustering = {'label':[], 'raw':[], 'center':[]}
        self.test_metrics_best = []
        self.con_mats = []
        self.server_model = model


    def aggregate(self, model_list, uploaded_weights, algorithm='FedAvg'):
        # update server model with FedAvg algorithm
        # model_list: list of models
        # set the server model to 0
        with torch.no_grad():
            for name, param in self.server_model.named_parameters():
                param.data.zero_()
        if algorithm == 'Fedprox':
            uploaded_weights = torch.ones(len(model_list))/len(model_list)
            
        for w, client_model in zip(uploaded_weights, model_list):
            for server_param, client_param in zip(self.server_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data * w
        
        return self.server_model
    
    def acc(self, nodes, weight_list):
        global_test_metrics = [0]*2
        for i in range(len(weight_list)):
            for j in range(len(global_test_metrics)):
                global_test_metrics[j] += weight_list[i]*nodes[i].test_metrics[-1][j]
        print('GLOBAL Accuracy, Macro F1 is %.2f %%, %.2f %%' % (100*global_test_metrics[0], 100*global_test_metrics[1]))
        self.test_metrics.append(global_test_metrics)

class node(object):
    def __init__(self, device, train_data, test_data, model, objective, optimizer, id,algorithm='FedAvg'):
        self.device = device
        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.id = id
        self.data_size = len(train_data)
        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []
        self.test_metrics = []
        self.clustering = {'label':[], 'raw':[], 'center':[]}
        self.test_metrics_best = []
        self.con_mats = []
        self.criterion = nn.CrossEntropyLoss()
        self.algorithm = algorithm

    def train(self, local_epochs,mu=0.1):
        model = self.model.to(self.device)
        model.to(self.device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4,betas=(0.9, 0.999))
        model.train()
        loss = 0
        for epoch in range(local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_data):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = model(data)
                loss = self.criterion(output, target)
                if self.algorithm == 'Fedprox':
                    # for name, param in model.named_parameters():
                    #         loss += self.args.mu * torch.norm(param - self.server_model.state_dict()[name]) ** 2/2
                    for w, w_t in zip(model.parameters(), self.server_model.parameters()):
                        loss += mu/2*(w - w_t).norm(2)**2

                loss.backward()
                self.optimizer.step()
                loss += loss.item()
        return model, loss / local_epochs
    
    def test(self, model):
        model.eval()
        test_loss = 0
        correct = 0
        test_metrics = [0]*2
        label_ts = []
        pred_ts = []
        with torch.no_grad():
            for data, target in self.test_data:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                label_ts.extend(target.cpu().numpy())
                pred_ts.extend(pred.cpu().numpy())
        test_loss /= len(self.test_data.dataset)
        test_acc = 100. * correct / len(self.test_data.dataset)
        print('local accuracy:',test_acc)
        test_metrics[0] = test_acc
        test_metrics[1] = self.f1_score(label_ts, pred_ts, average='macro')
        self.test_metrics.append(test_metrics)
        self.label_ts = label_ts
        self.pred_ts = pred_ts
        self.test_loss.append(test_loss)
        self.test_acc.append(test_acc)
        return test_loss, test_acc, test_metrics

    
num_nodes = 20 #4
dataset_name = 'data/jammer_split/client20_dir01/'
train_splited, test_splited, split_para = loader_gnss.load_dataloader(dataset_name, num_nodes)
model = CNNJammer()

train_splited_loader = [torch.utils.data.DataLoader(train_splited[i], batch_size=64, shuffle=True) for i in range(num_nodes)]
test_splited_loader = [torch.utils.data.DataLoader(test_splited[i], batch_size=64, shuffle=False) for i in range(num_nodes)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.5)
objective = nn.CrossEntropyLoss()
server_class = server(model, device)
nodes = [node(device, train_splited_loader[i], test_splited_loader[i], model, objective, optimizer, i) for i in range(num_nodes)]
local_epochs = 10
epochs =100
weights_list = [len(train_splited_loader[i].dataset)/sum([len(train_splited_loader[i].dataset) for i in range(num_nodes)]) for i in range(num_nodes)]
for epoch in range(epochs):
    for i in range(num_nodes):
        nodes[i].model = copy.deepcopy(server_class.server_model)
        _,loss = nodes[i].train(local_epochs)
        print('loss:',loss)
        print('node %d local train done' % (i))
    server_class.aggregate([nodes[i].model for i in range(num_nodes)], weights_list)
    # assign the server model to each node

    for i in range(num_nodes):
        nodes[i].test(server_class.server_model)
    server_class.acc(nodes, weights_list)


