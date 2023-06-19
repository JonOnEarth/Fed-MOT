import torch
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas.plotting import parallel_coordinates
import copy

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
                
                elif aggregated_method == 'AA':
                    for w, idx in zip(weight_list, range(len(model_list))):
                        new_param[name] += w * model_list[idx].state_dict()[name].to(self.device)
                    for w, idx in zip(weight_list, range(len(model_list))):
                        new_lambda[name] += w * (1/model_lambda_list[idx][name] + model_list[idx].state_dict()[name]**2 - new_param[name]**2)
                        # check if element in new_lambda is negative, change it to eps
                    new_lambda[name] = torch.where(new_lambda[name] < 0, torch.full_like(new_lambda[name], eps), new_lambda[name])
        return new_param, new_lambda

    
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

    def weighted_clustering_GNN(self, nodes, idlist, num_clusters):
        pass

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
