import torch
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas.plotting import parallel_coordinates
import copy
import random
# import skfuzzy as fuzz
# from skfuzzy import control as ctrl
from sklearn.metrics import confusion_matrix
from sklearn.mixture import BayesianGaussianMixture
from fedbase.server.dirichlet_process import DPMM

class server_class():
    def __init__(self, device):
        self.device = device
        self.test_metrics = []
        self.clustering = {'label':[], 'raw':[], 'center':[]}
        self.test_metrics_best = []
        self.con_mats = []

    def assign_model(self, model):
        try:
            self.model.load_state_dict(model.state_dict())
        except:
            self.model = model
        self.model.to(self.device)

    def assign_model_lambda(self, model_lambda):
        self.model_lambda = model_lambda

    # def aggregate(self, model_list, weight_list):
    #     aggregated_weights = model_list[0].state_dict()
    #     for j in aggregated_weights.keys():
    #         aggregated_weights[j] = torch.zeros(aggregated_weights[j].shape).to(self.device)
    #     # sum_size = sum([nodes[i].data_size for i in idlist])
    #     for i in range(len(model_list)):
    #         for j in model_list[i].state_dict().keys():
    #             aggregated_weights[j] += model_list[i].state_dict()[j]*weight_list[i]
    #     return aggregated_weights
    
    def aggregate(self, model_list, weight_list):
        new_param = {}
        new_model = copy.deepcopy(model_list[0])
        with torch.no_grad():
            for name, param in new_model.named_parameters():
                param.data.zero_()

        for w, client_model in zip(weight_list, model_list):
            for new_model_param, model_param in zip(new_model.parameters(), client_model.parameters()):
                new_model_param.data += w * model_param.data.to(self.device)
        return new_model.state_dict()
    
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
                    weight_list_new = [1/len(model_list)] * len(model_list)
                    for w, idx in zip(weight_list_new, range(len(model_list))):
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

    def acc2(self, nodes, weight_list,type=None):
        global_test_metrics = [0]*2
        for i in range(len(weight_list)):
            for j in range(len(global_test_metrics)):
                global_test_metrics[j] += weight_list[i]*nodes[i].test_metrics[-1][j]
        print('GLOBAL Accuracy, Macro F1 is %.2f %%, %.2f %%' % (100*global_test_metrics[0], 100*global_test_metrics[1]))
        if type=='save':
            self.test_metrics_best.append(global_test_metrics)
        return global_test_metrics

    def acc(self, nodes, weight_list):
        global_test_metrics = [0]*2
        for i in range(len(weight_list)):
            for j in range(len(global_test_metrics)):
                global_test_metrics[j] += weight_list[i]*nodes[i].test_metrics[-1][j]
        print('GLOBAL Accuracy, Macro F1 is %.2f %%, %.2f %%' % (100*global_test_metrics[0], 100*global_test_metrics[1]))
        self.test_metrics.append(global_test_metrics)

    def acc_conf(self, nodes, weight_list):
        global_test_metrics = [0]*2
        # global_con_mat = np.zeros((10,10))
        label_ts_all = []
        pred_ts_all = []
        for i in range(len(weight_list)):
            for j in range(len(global_test_metrics)):
                    global_test_metrics[j] += weight_list[i]*nodes[i].test_metrics[-1][j]
            label_ts_all.extend(nodes[i].label_ts)
            pred_ts_all.extend(nodes[i].predict_ts)
        print('GLOBAL Accuracy, Macro F1 is %.2f %%, %.2f %%' % (100*global_test_metrics[0], 100*global_test_metrics[1]))
        self.test_metrics.append(global_test_metrics)
        self.con_mats.append(confusion_matrix(label_ts_all, pred_ts_all))

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

    def weighted_clustering_unknown(self, nodes, idlist):
        
        X = []
        for i in idlist:
            X.append(np.array(torch.flatten(nodes[i].model.state_dict()[list(nodes[i].model.state_dict().keys())[-2]]).cpu()))
        # use dirichlet process to cluster the nodes
        # dpgmm = BayesianGaussianMixture(n_components=10, covariance_type='diag',weight_concentration_prior=0.002, warm_start=True).fit(X)
        dpgm = DPMM(alpha=0.1, covariance_type='same', n_iter=100)
        dpgm.fit(np.asarray(X))
        labels = dpgm.predict(np.asarray(X))
        print(labels)
        # rewrite the label to 0,1,2,3...
        # Unique labels sorted
        unique_labels = np.unique(labels)
        # Create a dictionary mapping original labels to new labels (0, 1, 2, 3...)
        label_mapping = {label: index for index, label in enumerate(unique_labels)}
        # Map the original labels to new labels
        new_labels = [label_mapping[label] for label in labels]
        K = len(set(new_labels))
        print("K:",K, "labels:",new_labels)
        print([list(labels).count(i) for i in range(K)])
        for i in idlist:
            nodes[i].label = labels[i]
        self.clustering['label'].append(list(labels.astype(int)))
        return K

    
    def soft_clustering(self, nodes, idlist, K, weight_type='data_size'):
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

    def fuzzy_clustering(self, model_list, K, xii=0.9, sigma=10, type='cosine', weight_list=None):
        # model_list is a list of models
        # return the kmeans result of the model_list
        # get the parameters of the model_list
        # type: local, samples, cosine, distance, equal
        X = []
        for model in model_list:
            para = []
            for name, param in model.named_parameters():
                para.append(torch.flatten(param.cpu()).detach())
            X.append(torch.concatenate(para))
        
        if K == len(model_list):
            labels = range(K)
            if type == 'local':
                new_model_list = model_list
            else:
                similarity_matrix= torch.zeros(K,K)
                for i in range(K):
                    for j in range(K):
                        # if i != j:
                        similarity_matrix[i,j] = self.get_weights(X[i],X[j],type=type,sigma=sigma,weight=weight_list[j])
                xi = torch.zeros(K,K)
                for i in range(K):
                    if type == 'cosine' or type == 'distance':
                        if xii:
                            similarity_matrix[i,i] = 0
                            for j in range(K):
                                xi[i,j] = (1-xii)* similarity_matrix[i,j]/torch.sum(similarity_matrix[i,:])
                            xi[i,i] = xii
                        else:
                            xi[i,:] = similarity_matrix[i,:]/torch.sum(similarity_matrix[i,:])
                    elif type == 'samples' or type== 'equal':
                        for j in range(K):
                            xi[i,j] = weight_list[j]/torch.sum(torch.Tensor(weight_list))

                new_model_list = []

                for i in range(K):
                    # cluster_model = copy.deepcopy(model_list[i])
                    # for name, param in model_list[0].named_parameters():
                    #     cluster_model.state_dict()[name] = torch.zeros(param.shape)
                    #     for j in range(K):
                    #         cluster_model.state_dict()[name] += model_list[j].state_dict()[name]*xi[i,j]
                    # new_model_list.append(cluster_model)
                    new_param = {}
                    with torch.no_grad():
                        for name, param in model_list[i].named_parameters():
                            new_param[name] = param.data.zero_()
                            for w, idx in zip(xi[i], range(len(model_list))):
                                new_param[name] += w * model_list[idx].state_dict()[name].to(self.device)
                    self.distribute([model_list[i]], new_param)
                    new_model_list.append(model_list[i])

        elif K < len(model_list):
            # Apply fuzzy c-means clustering
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                X, K, 2, error=0.005, maxiter=1000, init=None
            )
            # Predict cluster membership for each data point
            labels = np.argmax(u, axis=0)
            new_model_list = []
            # cntr include the center of each cluster
            # how to make cntr like the model_list
            for i in range(K):
                pass

        return new_model_list, labels

            
    def get_weights(self, x,y, type='cosine',sigma=1, weight=None):
        if type == 'cosine':
            similarity = torch.exp(sigma*torch.cosine_similarity(x,y,dim=0))
        elif type == 'distance':
            similarity = torch.exp(-sigma*torch.norm(x-y))
        elif type == 'equal':
            similarity = 1
        elif type == 'samples':
            similarity = weight
        else: 
            raise ValueError('type should be cosine or distance')
        return similarity

    def aggregate_amp(self, model_list, xii=0.5, sigma=10, type='cosine', weight_list=None):
        new_model_list = []
        X = []
        for model in model_list:
            para = []
            for name, param in model.named_parameters():
                para.append(torch.flatten(param.cpu()).detach())
            X.append(torch.cat(para))

        for c in range(len(model_list)):
            mu = copy.deepcopy(model_list[c])
            for param in mu.parameters():
                param.data.zero_()

            similarity = torch.zeros(len(model_list))
            for j, mw in enumerate(model_list):
                similarity[j] = self.get_weights(X[c],X[j],type=type,sigma=sigma,weight=weight_list[j])

            if xii:
                similarity[c] = 0
                similarity = similarity/torch.sum(similarity)
                xi = (1-xii)* similarity
                xi[c] = xii
            else:
                xi = similarity/torch.sum(similarity)

            for j, mw in enumerate(model_list):
                for param, param_j in zip(mu.parameters(), mw.parameters()):
                    param.data += xi[j]*param_j.data
            new_model_list.append(mu)
            labels = range(len(model_list))
        return new_model_list, labels
    
    # for Fedsoft aggregation
    def aggregate_fedsoft(self, selection, K,nodes,cluster_vec,do_selection=True):
        for s in range(K):
            next_weights = self.generate_zero_weights()
            for k in selection[s]:
                if do_selection:
                    aggregation_weight = 1. / K
                else:
                    aggregation_weight = self.importance_weights_matrix[k][s]
                client_weights = nodes[k].model.state_dict()
                for key in next_weights.keys():
                    next_weights[key] += aggregation_weight * client_weights[key].cpu()
            cluster_vec[s].load_state_dict(state_dict=next_weights)
            cluster_vec[s].to(self.device)
        return cluster_vec

    def generate_zero_weights(self):
        self._zero_weights = {}
        for key, val in self.model.state_dict().items():
            self._zero_weights[key] = torch.zeros(size=val.shape, dtype=torch.float32)
        return copy.deepcopy(self._zero_weights)
    
    def fedem_acc(self, nodes, weight_list, num_learners):
        """
        Calculate and print global accuracy for FedEM
        
        Args:
            nodes: List of client nodes
            weight_list: Weights for aggregating client metrics
            num_learners: Number of learners per client
        """
        global_test_metrics = [0] * 2  # [accuracy, f1_score]
        
        for i in range(len(weight_list)):
            if nodes[i].test_metrics:  # Check if test metrics exist
                for j in range(len(global_test_metrics)):
                    global_test_metrics[j] += weight_list[i] * nodes[i].test_metrics[-1][j]
        
        print(f'GLOBAL Accuracy, Macro F1 is {100 * global_test_metrics[0]:.2f} %, {100 * global_test_metrics[1]:.2f} %')
        self.test_metrics.append(global_test_metrics)