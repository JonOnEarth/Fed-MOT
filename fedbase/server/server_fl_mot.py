import torch
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas.plotting import parallel_coordinates
import copy
import random


class server_class():
    def __init__(self, device):
        self.device = device
        self.test_metrics = []
        self.clustering = {'label':[], 'raw':[], 'center':[]}

    def assign_model(self, model):
        try:
            self.model.load_state_dict(model.state_dict())
        except:
            self.model = model
        self.model.to(self.device)

    def assign_model_lambda(self, model_lambda):
        self.model_lambda = model_lambda

    def aggregate(self, model_list, weight_list):
        aggregated_weights = model_list[0].state_dict()
        for j in aggregated_weights.keys():
            aggregated_weights[j] = torch.zeros(aggregated_weights[j].shape).to(self.device)
        # sum_size = sum([nodes[i].data_size for i in idlist])
        for i in range(len(model_list)):
            for j in model_list[i].state_dict().keys():
                aggregated_weights[j] += model_list[i].state_dict()[j]*weight_list[i]
        return aggregated_weights
    
    def aggregate_bayes(self, model_list, model_lambda_list, weight_list, aggregated_method='GA',eps=0):
        # update the server model
        new_param = {}
        new_lambda = {}
        model_example = copy.deepcopy(model_list[0])
        model_lambda_example = copy.deepcopy(model_lambda_list[0])
        with torch.no_grad():
            for name, param in model_example.named_parameters():
                new_param[name] = param.data.zero_()
                new_lambda[name] = model_lambda_example[name].data.zero_()
                if aggregated_method == 'GA':
                    for w, idx in zip(weight_list, range(len(model_list))):
                        new_param[name] += w * model_lambda_list[idx][name] * model_list[idx].state_dict()[name].to(self.device)
                        new_lambda[name] += w * model_lambda_list[idx][name]
                    # new_param[name] -= (sum(weight_list) - 1) * self.server_lambda[name] * \
                    #                 self.server_model.state_dict()[name].to(device)
                    # new_lambda[name] -= (sum(weight_list) - 1) * self.server_lambda[name]
                    new_param[name] /= (new_lambda[name] + eps)
                    new_weight = torch.prod(weight_list)
                elif aggregated_method == 'AA':
                    for w, idx in zip(weight_list, range(len(model_list))):
                        new_param[name] += w * model_list[idx].state_dict()[name].to(self.device)
                    for w, idx in zip(weight_list, range(len(model_list))):
                        new_lambda[name] += w * (1/model_lambda_list[idx][name] + model_list[idx].state_dict()[name]**2 - new_param[name]**2)
                        # check if element in new_lambda is negative, change it to eps
                    new_lambda[name] = torch.where(new_lambda[name] < 0, torch.full_like(new_lambda[name], eps), new_lambda[name])
                    new_weight = torch.sum(weight_list)
        return new_param, new_lambda, new_weight

    def aggregate_bayes_H(self, model_list, model_lambda_list, weight_list, H, aggregated_method='GA',eps=0, weight_type='equal'):
        # def a sub function for kmenas of model_list's parameters
        def kmeans_model_list(model_list, weight_list, H):
            # model_list is a list of models, weight_list is the weight of each model, H is the number of clusters
            # return the kmeans result of the model_list
            # get the parameters of the model_list
            X = []
            for model in model_list:
                X.append(np.array(torch.flatten(model.state_dict()[list(model.state_dict().keys())[-2]].cpu())))
            kmeans = KMeans(n_clusters=H, random_state=0).fit(np.asarray(X), sample_weight= weight_list)
            labels = kmeans.labels_
            return labels
        # update the server model
        new_param = {}
        new_lambda = {}
        model_example = copy.deepcopy(model_list[0])
        model_lambda_example = copy.deepcopy(model_lambda_list[0])
        if weight_type == 'equal':
            weight_list = [1/len(model_list)] * len(model_list)
            
        labels = kmeans_model_list(model_list, weight_list, H)
        new_params, new_lambdas, new_weights = [], [], []
        for h in range(H):
            model_list_h = []
            model_lambda_list_h = []
            weight_list_h = []
            for idx in range(len(model_list)):
                if labels[idx] == h:
                    model_list_h.append(model_list[idx])
                    model_lambda_list_h.append(model_lambda_list[idx])
                    weight_list_h.append(weight_list[idx])
            with torch.no_grad():
                for name, param in model_example.named_parameters():
                    new_param[name] = param.data.zero_()
                    new_lambda[name] = model_lambda_example[name].data.zero_()
                    if aggregated_method == 'GA':
                        for w, idx in zip(weight_list_h, range(len(model_list_h))):
                            new_param[name] += w * model_lambda_list_h[idx][name] * model_list_h[idx].state_dict()[name].to(self.device)
                            new_lambda[name] += w * model_lambda_list_h[idx][name]
                        # new_param[name] -= (sum(weight_list) - 1) * self.server_lambda[name] * \
                        #                 self.server_model.state_dict()[name].to(device)
                        # new_lambda[name] -= (sum(weight_list) - 1) * self.server_lambda[name]
                        new_param[name] /= (new_lambda[name] + eps)
                    
                    elif aggregated_method == 'AA':
                        for w, idx in zip(weight_list_h, range(len(model_list_h))):
                            new_param[name] += w * model_list_h[idx].state_dict()[name].to(self.device)
                        for w, idx in zip(weight_list_h, range(len(model_list_h))):
                            new_lambda[name] += w * (1/model_lambda_list_h[idx][name] + model_list_h[idx].state_dict()[name]**2 - new_param[name]**2)
                            # check if element in new_lambda is negative, change it to eps
                        new_lambda[name] = torch.where(new_lambda[name] < 0, torch.full_like(new_lambda[name], eps), new_lambda[name])
            new_params.append(new_param)
            new_lambdas.append(new_lambda)
            new_weights.append(sum(weight_list_h))
        return new_params, new_lambdas, new_weights
    
    def distribute(self, model_in_list, model_dis_dict = None):
        if not model_dis_dict:
            model_dis_dict = self.model.state_dict()
        for i in model_in_list:
            # i.load_state_dict(model_dis_dict)
            for name, param in i.named_parameters():
                i.state_dict()[name].data.copy_(model_dis_dict[name])  # https://discuss.pytorch.org/t/how-can-i
                # -modify-certain-layers-weight-and-bias/11638
                # self.server_lambda[name] = new_lambda[name]

    def distribute_lambda(self, model_lambda_in_list, model_lambda_dis_dict = None):
        if not model_lambda_dis_dict:
            model_lambda_dis_dict = self.model_lambda
        for i in model_lambda_in_list:
            i = model_lambda_dis_dict
            # for name, param in i.named_parameters():
            #     # i.state_dict()[name].data.copy_(model_dis_dict[name])  # https://discuss.pytorch.org/t/how-can-i
            #     # # -modify-certain-layers-weight-and-bias/11638
            #     i[name] = model_lambda_dis_dict[name]

    def acc(self, nodes, weight_list):
        global_test_metrics = [0]*2
        for i in range(len(weight_list)):
            for j in range(len(global_test_metrics)):
                global_test_metrics[j] += weight_list[i]*nodes[i].test_metrics[-1][j]
        print('GLOBAL Accuracy, Macro F1 is %.2f %%, %.2f %%' % (100*global_test_metrics[0], 100*global_test_metrics[1]))
        self.test_metrics.append(global_test_metrics)

    def client_sampling(self, frac, distribution):
        pass

    def test(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = torch.flatten(labels)
                labels = labels.to(self.device, dtype = torch.long)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy on the %d test cases: %.2f %%' % (total, 100*correct / total))

    def weighted_clustering(self, nodes, idlist, K, weight_type='data_size'):
        weight = []
        X = []
        sum_size = sum([nodes[i].data_size for i in idlist])
        # print(list(nodes[0].model.state_dict().keys()))
        for i in idlist:
            if weight_type == 'equal':
                weight.append(1/len(idlist))
            elif weight_type == 'data_size':
                weight.append(nodes[i].data_size/sum_size)
            elif weight_type == 'loss':
                weight.append(nodes[i].weight)
            X.append(np.array(torch.flatten(nodes[i].model.state_dict()[list(nodes[i].model.state_dict().keys())[-2]]).cpu()))
        # print(X, np.array(X).shape)
        # normalized weight
        weight = torch.tensor(weight)/torch.sum(torch.tensor(weight))
        kmeans = KMeans(n_clusters=K, n_init = 5).fit(np.asarray(X), sample_weight= weight)
        labels = kmeans.labels_
        print(labels)
        print([list(labels).count(i) for i in range(K)])
        for i in idlist:
            nodes[i].label = labels[i]
        self.clustering['label'].append(list(labels.astype(int)))
        # self.clustering['raw'].append(X)
        # self.clustering['center'].append(kmeans.cluster_centers_)

    def sample_nodes(self, nodes, sampling_rate,sample_with_replacement):
        n_nodes = max(1, int(len(nodes)*sampling_rate))
        if sample_with_replacement:
            return random.Random.choices(
                    population=nodes,
                    weights=[node.weight for node in nodes],
                    k=n_nodes,
                )
        else:
            return random.Random.sample(
                    population=nodes,
                    k=n_nodes,
                )

    def clustering_plot(self):
        # print(self.clustering)
        # self.clustering =[[1,1,2,2,3,3],[1,1,1,2,2,2],[1, 1, 1, 2, 2, 2],[1, 1, 1, 2, 2, 2]]
        col = [str(i) for i in range(len(self.clustering))]+['id']
        self.clustering.append(list(range(len(self.clustering[0]))))
        data= pd.DataFrame(np.array(self.clustering).T,columns= col)
        for i in data.columns:
            data[i]=data[i].apply(lambda x: str(x))
        # Make the plot
        parallel_coordinates(data, 'id', colormap=plt.get_cmap("Set2"))
        plt.show()
    # local prune function based on the rs

    def prune_local(self, rs_client, left_num =1):
        
        rs_sort, idex = torch.sort(rs_client, descending=True)
            # if idex is 0-dim tensor
        if idex.dim() == 0:
            idex = idex.unsqueeze(0)
        return idex[:left_num] if idex[:left_num].dim() == 1 else idex[:left_num].squeeze(1)
