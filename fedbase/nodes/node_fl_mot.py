import torch
from torch import linalg as LA
from fedbase.utils.model_utils import save_checkpoint, load_checkpoint
from fedbase.model.model import CNNMnist, MLP
from sklearn.metrics import accuracy_score, f1_score
from fedbase.utils.tools import unpack_args
from functools import partial
from statistics import mode
import torch.nn.functional as F
import copy
import numpy as np

from sklearn.metrics import confusion_matrix

class node():
    def __init__(self, id, device):
        self.id = id
        self.test_metrics = []
        self.step = 0
        self.device = device
        self.grads = []
        # self.con_mats = []
        self.label_ts = []
        self.predict_ts = []
        self.importance_estimated = []

    def assign_train(self, data):
        self.train = data
        self.data_size = len(data.dataset)
    
    def assign_test(self,data):
        self.test = data

    def assign_model(self, model):
        try:
            self.model.load_state_dict(model.state_dict())
        except:
            self.model = model
        self.model.to(self.device)

    def assign_model_lambda(self, model_lambda):
        self.model_lambda = model_lambda

    def assign_objective(self, objective):
        self.objective = objective

    def assign_optim(self, optim):
        self.optim = optim

    def local_update_steps(self, local_steps, train_single_step_func):
        # print(len(self.train), self.step)
        if len(self.train) - self.step > local_steps:
            for k, (inputs, labels) in enumerate(self.train):
                if k < self.step or k >= self.step + local_steps:
                    continue
                train_single_step_func(inputs, labels)
            self.step = self.step + local_steps
        else:
            for k, (inputs, labels) in enumerate(self.train):
                if k < self.step:
                    continue
                train_single_step_func(inputs, labels)
            for j in range((local_steps-len(self.train)+self.step)//len(self.train)):
                for k, (inputs, labels) in enumerate(self.train):
                    train_single_step_func(inputs, labels)
            for k, (inputs, labels) in enumerate(self.train):
                if k >=(local_steps-len(self.train)+self.step)%len(self.train):
                    continue
                train_single_step_func(inputs, labels)
            self.step = (local_steps-len(self.train)+self.step)%len(self.train)

    def local_update_epochs(self, local_epochs, train_single_step_func):
        # local_steps may be better!!
        running_loss = 0
        for j in range(local_epochs):
            for k, (inputs, labels) in enumerate(self.train):
                train_single_step_func(inputs, labels)

    def train_single_step(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels = torch.flatten(labels)
        labels = labels.to(self.device, dtype = torch.long)
        # print(labels)
        # zero the parameter gradients
        # self.model.zero_grad(set_to_none=True)
        self.optim.zero_grad()
        # forward + backward + optimize
        with torch.cuda.amp.autocast():
            outputs = self.model(inputs)
            # optim
            loss = self.objective(outputs, F.one_hot(labels, outputs.shape[1]).float())
        loss.backward()

        # calculate accumulate gradients
        # grads = torch.tensor([])
        # for index, param in enumerate(self.model.parameters()):
        #     # param.grad = torch.tensor(grads[index])
        #     grads= torch.cat((grads, torch.flatten(param.grad).cpu()),0)
        # self.grads.append(grads)

        self.optim.step()
        # self.train_steps+=1

        # print('after', self.objective(self.model(inputs), labels))
        
    # for fedprox and ditto
    def train_single_step_fedprox(self, inputs, labels, reg_lam = None, reg_model = None):
        inputs = inputs.to(self.device)
        labels = torch.flatten(labels)
        labels = labels.to(self.device, dtype = torch.long)
        # zero the parameter gradients
        # self.model.zero_grad(set_to_none=True)
        self.optim.zero_grad()
        # forward + backward + optimize
        outputs = self.model(inputs)
        if reg_lam:
            reg_model.to(self.device)
            reg = torch.square(torch.norm(torch.cat(tuple([torch.flatten(self.model.state_dict()[k] - reg_model.state_dict()[k])\
                    for k in self.model.state_dict().keys()]),0),2))
        else:
            reg, reg_lam = 0, 0
        self.loss = self.objective(outputs, labels) + reg_lam*reg/2
        # print(self.objective(outputs, labels))
        self.loss.backward()
        self.optim.step()
        # print('after', self.objective(self.model(inputs), labels))
    
    def train_single_step_bayes(self, inputs, labels, csd_importance=1, clip=10, reg_model = None, reg_model_lambda=None):  # client_update

        new_lambda = dict()
        new_mu = dict()
        # log_ce_loss = 0
        # log_csd_loss = 0
        # model_state_dict = self.model.state_dict()
        reg_model.to(self.device)
        for name, param in self.model.named_parameters():
            new_lambda[name] = copy.deepcopy(reg_model_lambda[name])
            new_mu[name] = copy.deepcopy(reg_model.state_dict()[name])

        inputs = inputs.to(self.device)
        labels = torch.flatten(labels)
        labels = labels.to(self.device, dtype = torch.long)

        self.optim.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = self.model(inputs)
            # ce_loss = self.criterion(output, target)
            ce_loss = self.objective(outputs, F.one_hot(labels, outputs.shape[1]).float())
            csd_loss = self.get_csd_loss(self.model, new_mu, new_lambda) if csd_importance > 0 else 0
        ce_loss.backward(retain_graph=True)

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.model_lambda[name] +=  (len(labels) / 
                                self.data_size) * param.grad.data.clone() ** 2

        self.optim.zero_grad()
        loss = ce_loss + csd_importance * csd_loss
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
        self.optim.step()

        # log_ce_loss += ce_loss.item()
        # log_csd_loss += csd_loss.item() if self.args.csd_importance > 0 else 0
        # self.ce_loss = ce_loss.item()
        # return model, model_lambda, log_ce_loss

    def get_ce_loss(self, temperature=torch.tensor(10.0)):
        ce_losss = 0
        num = 0
        for k, (inputs, labels) in enumerate(self.train):
            inputs = inputs.to(self.device)
            labels = torch.flatten(labels)
            labels = labels.to(self.device, dtype = torch.long)
            outputs = self.model(inputs)
            ce_loss = self.objective(outputs, F.one_hot(labels, outputs.shape[1]).float())
            ce_losss += ce_loss.item()
            num += len(inputs)
        # get ce loss of all data in train and average
        ce_loss_weight = ce_losss/num

        self.weight = torch.exp(-ce_loss_weight*temperature)
        return ce_losss
    # for IFCA
    def local_train_loss(self, model):
        model.to(self.device)
        train_loss = 0
        i = 0
        with torch.no_grad():
            for data in self.train:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = torch.flatten(labels)
                labels = labels.to(self.device, dtype = torch.long)
                # forward
                outputs = model(inputs)
                train_loss += float(self.objective(outputs, labels))
                i+=1
                if i>=10:
                    break
        # return train_loss/len(self.train)
        return train_loss/i
    
    # for FedSoft estimate importance weights
    def estimate_importance_weights(self,cluster_vec,num_classes,count_smoother=0.0001):
        
        with torch.no_grad():
            table = np.zeros((len(cluster_vec), self.data_size))
            start_idx = 0
            nst_cluster_sample_count = [0] * len(cluster_vec)
            for data in self.train:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = torch.flatten(labels)
                labels = labels.to(self.device, dtype = torch.long)
                for s, cluster in enumerate(cluster_vec):
                    cluster.eval()
                    cluster.to(self.device)
                    out = cluster(inputs)
                    out = out.view(-1, num_classes)
                    loss = torch.nn.CrossEntropyLoss(reduction='none')(out, labels)
                    table[s][start_idx:start_idx + len(inputs)] = loss.cpu()
                start_idx += len(inputs)

            min_loss_idx = np.argmin(table, axis=0)
            for s in range(len(cluster_vec)):
                nst_cluster_sample_count[s] += np.sum(min_loss_idx == s)
            for s in range(len(cluster_vec)):
                if nst_cluster_sample_count[s] == 0:
                    nst_cluster_sample_count[s] = count_smoother * self.data_size
            self.importance_estimated = np.array([1.0 * nst / self.data_size for nst in nst_cluster_sample_count])
    
    # for FedSoft Local train
    def train_single_step_fedsoft(self, inputs, labels, cluster_vec=None, reg_lam = None):
        inputs = inputs.to(self.device)
        labels = torch.flatten(labels)
        labels = labels.to(self.device, dtype = torch.long)
        self.optim.zero_grad()
        out = self.model(inputs)
        self.loss = self.objective(out, labels)
        mse_loss = torch.nn.MSELoss(reduction='sum')
        for i, cluster in enumerate(cluster_vec):
            cluster.to(self.device)
            l2 = None
            for (param_local, param_cluster) in zip(self.model.parameters(), cluster.parameters()):
                if l2 is None:
                    l2 = mse_loss(param_local, param_cluster)
                else:
                    l2 += mse_loss(param_local, param_cluster)
            self.loss += reg_lam / 2 * self.importance_estimated[i] * l2
        self.loss.backward()
        self.optim.step()

    # for FedSoft get importance
    def get_importance(self, count=True):
        if count:
            return [ust * self.data_size for ust in self.importance_estimated]
        else:
            return self.importance_estimated

    def local_train_acc(self, model):
        model.to(self.device)
        predict_ts = torch.empty(0).to(self.device)
        label_ts = torch.empty(0).to(self.device)
        i = 0
        with torch.no_grad():
            for data in self.train:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = torch.flatten(labels)
                labels = labels.to(self.device, dtype = torch.long)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predict_ts = torch.cat([predict_ts, predicted], 0)
                label_ts = torch.cat([label_ts, labels], 0)
                i+=1
                if i>=10:
                    break
        acc = accuracy_score(label_ts.cpu(), predict_ts.cpu())
        return acc

    def local_test(self, model_res = None):
        predict_ts = torch.empty(0).to(self.device)
        label_ts = torch.empty(0).to(self.device)
        with torch.no_grad():
            for data in self.test:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = torch.flatten(labels)
                labels = labels.to(self.device, dtype = torch.long)
                if model_res:
                    model_res.to(self.device)
                    outputs = model_res(inputs) + self.model(inputs)
                else:
                    outputs = self.model(inputs)
                # print(outputs.data.dtype)
                _, predicted = torch.max(outputs.data, 1)
                predict_ts = torch.cat([predict_ts, predicted], 0)
                label_ts = torch.cat([label_ts, labels], 0)
        acc = accuracy_score(label_ts.cpu(), predict_ts.cpu())
        macro_f1 = f1_score(label_ts.cpu(), predict_ts.cpu(), average='macro')
        # con_mat = confusion_matrix(label_ts.cpu().numpy(), predict_ts.cpu().numpy())
        # micro_f1 = f1_score(label_ts.cpu(), predict_ts.cpu(), average='micro')
        # print('Accuracy, Macro F1, Micro F1 of Device %d on the %d test cases: %.2f %%, %.2f, %.2f' % (self.id, len(label_ts), acc*100, macro_f1, micro_f1))
        print('Accuracy, Macro F1 of Device %d on the %d test cases: %.2f %%, %.2f %%' % (self.id, len(label_ts), acc*100, macro_f1*100))
        self.test_metrics.append([acc, macro_f1])

    def local_test_conf(self, model_res = None):
        predict_ts = torch.empty(0).to(self.device)
        label_ts = torch.empty(0).to(self.device)
        with torch.no_grad():
            for data in self.test:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = torch.flatten(labels)
                labels = labels.to(self.device, dtype = torch.long)
                if model_res:
                    model_res.to(self.device)
                    outputs = model_res(inputs) + self.model(inputs)
                else:
                    outputs = self.model(inputs)
                # print(outputs.data.dtype)
                _, predicted = torch.max(outputs.data, 1)
                predict_ts = torch.cat([predict_ts, predicted], 0)
                label_ts = torch.cat([label_ts, labels], 0)
        acc = accuracy_score(label_ts.cpu(), predict_ts.cpu())
        macro_f1 = f1_score(label_ts.cpu(), predict_ts.cpu(), average='macro')
        # con_mat = confusion_matrix(label_ts.cpu().numpy(), predict_ts.cpu().numpy())
        # micro_f1 = f1_score(label_ts.cpu(), predict_ts.cpu(), average='micro')
        # print('Accuracy, Macro F1, Micro F1 of Device %d on the %d test cases: %.2f %%, %.2f, %.2f' % (self.id, len(label_ts), acc*100, macro_f1, micro_f1))
        print('Accuracy, Macro F1 of Device %d on the %d test cases: %.2f %%, %.2f %%' % (self.id, len(label_ts), acc*100, macro_f1*100))
        self.test_metrics.append([acc, macro_f1])
        # self.con_mats.append(con_mat)
        self.label_ts = label_ts.cpu().numpy()
        self.predict_ts = predict_ts.cpu().numpy()
        # return con_mat

    def local_ensemble_test(self, model_list, voting = 'soft'):
        predict_ts = torch.empty(0).to(self.device)
        label_ts = torch.empty(0).to(self.device)
        if voting == 'max':
            # choose the model with the lowest loss
            node_loss = 1000
            m = 0
            for i,model in enumerate(model_list):
                node_loss_temp = self.local_train_loss(model)
                if node_loss > node_loss_temp:
                    node_loss = node_loss_temp
                    m = i
            model_list = [model_list[m]]
        with torch.no_grad():
            for data in self.test:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = torch.flatten(labels)
                labels = labels.to(self.device, dtype = torch.long)
                # out_hard = []
                if voting == 'soft' or voting == 'max':
                    out = torch.zeros(self.model(inputs).data.shape).to(self.device)
                    for model in model_list:
                        outputs = model(inputs)
                        out = out + outputs.data/len(model_list)
                    _, predicted = torch.max(out, 1)
                elif voting == 'hard':
                    out_hard = []
                    for model in model_list:
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        out_hard.append(predicted)       
                    predicted = torch.as_tensor([mode([out_hard[i][j] for i in range(len(out_hard))]) for j in range(len(out_hard[0]))]).to(self.device)
                    outputs = model_list[m](inputs)
                predict_ts = torch.cat([predict_ts, predicted], 0)
                label_ts = torch.cat([label_ts, labels], 0)
        acc = accuracy_score(label_ts.cpu(), predict_ts.cpu())
        macro_f1 = f1_score(label_ts.cpu(), predict_ts.cpu(), average='macro')
        # micro_f1 = f1_score(label_ts.cpu(), predict_ts.cpu(), average='micro')
        # print('Accuracy, Macro F1, Micro F1 of Device %d on the %d test cases: %.2f %%, %.2f, %.2f' % (self.id, len(label_ts), acc*100, macro_f1, micro_f1))
        print('Accuracy, Macro F1 of Device %d on the %d test cases: %.2f %%, %.2f %%' % (self.id, len(label_ts), acc*100, macro_f1*100))
        self.test_metrics.append([acc, macro_f1])

    def local_ensemble_test2(self, model_list, weight_list = None, voting = 'soft'):
        predict_ts = torch.empty(0).to(self.device)
        label_ts = torch.empty(0).to(self.device)
        with torch.no_grad():
            for data in self.test:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = torch.flatten(labels)
                labels = labels.to(self.device, dtype = torch.long)
                out_hard = []
                if voting == 'soft':
                    out = torch.zeros(self.model(inputs).data.shape).to(self.device)
                    for model in model_list:
                        outputs = model(inputs)
                        out = out + outputs.data/len(model_list)
                    _, predicted = torch.max(out, 1)
                elif voting == 'hard':
                    out_hard = []
                    for model in model_list:
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        out_hard.append(predicted)       
                    predicted = torch.as_tensor([mode([out_hard[i][j] for i in range(len(out_hard))]) for j in range(len(out_hard[0]))]).to(self.device)
                elif voting == 'product_weighted':
                    out = torch.zeros(self.model(inputs).data.shape).to(self.device)
                    for i in range(len(model_list)):
                        model = model_list[i]
                        outputs = model(inputs)
                        out = out + outputs.data*weight_list[i]
                    _, predicted = torch.max(out, 1)
                elif voting == 'sum_weighted':
                    out_pro = torch.zeros(self.model(inputs).data.shape).to(self.device)
                    for i in range(len(model_list)):
                        model = model_list[i]
                        outputs = model(inputs)
                        # output to probability
                        out_pro = out_pro + F.log_softmax(outputs, dim=1)+torch.log(weight_list[i]) #outs_pro.data*weight_list[i]
                    _, predicted = torch.max(out_pro, 1)
                predict_ts = torch.cat([predict_ts, predicted], 0)
                label_ts = torch.cat([label_ts, labels], 0)
        acc = accuracy_score(label_ts.cpu(), predict_ts.cpu())
        macro_f1 = f1_score(label_ts.cpu(), predict_ts.cpu(), average='macro')
        # micro_f1 = f1_score(label_ts.cpu(), predict_ts.cpu(), average='micro')
        # print('Accuracy, Macro F1, Micro F1 of Device %d on the %d test cases: %.2f %%, %.2f, %.2f' % (self.id, len(label_ts), acc*100, macro_f1, micro_f1))
        print('Accuracy, Macro F1 of Device %d on the %d test cases: %.2f %%, %.2f %%' % (self.id, len(label_ts), acc*100, macro_f1*100))
        self.test_metrics.append([acc, macro_f1])

    def get_csd_loss(self, model, mu, omega):
        loss_set = []
        for name, param in model.named_parameters():
            theta = model.state_dict()[name]
            # omega_dropout = torch.rand(omega[name].size()).cuda() if cuda else torch.rand(omega[name].size())
            # omega_dropout[omega_dropout>0.5] = 1.0
            # omega_dropout[omega_dropout <= 0.5] = 0.0
            # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
            mu[name] = mu[name].to(self.device)
            omega[name] = omega[name].to(self.device)
            loss_set.append((0.5) * (omega[name] * ((theta - mu[name]) ** 2)).sum())

        return sum(loss_set)