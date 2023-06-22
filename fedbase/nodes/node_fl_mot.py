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

class node():
    def __init__(self, id, device):
        self.id = id
        self.test_metrics = []
        self.step = 0
        self.device = device
        self.grads = []

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

    def train_single_step_bayes(self, inputs, labels, csd_importance=1, clip=10, reg_model = None, reg_model_lambda=None):  # client_update

        new_lambda = dict()
        new_mu = dict()
        # log_ce_loss = 0
        # log_csd_loss = 0
        # model_state_dict = self.model.state_dict()
        for name, param in self.model.named_parameters():
            new_lambda[name] = copy.deepcopy(reg_model_lambda[name])
            new_mu[name] = copy.deepcopy(reg_model.state_dict()[name])

        inputs = inputs.to(self.device)
        labels = torch.flatten(labels)
        labels = labels.to(self.device, dtype = torch.long)

        self.optim.zero_grad()
        outputs = self.model(inputs)
        # ce_loss = self.criterion(output, target)
        ce_loss = self.objective(outputs, F.one_hot(labels, outputs.shape[1]).float())
        csd_loss = get_csd_loss(self.model, new_mu, new_lambda) if csd_importance > 0 else 0
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

    def get_ce_loss(self, temperature):
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
                train_loss += self.objective(outputs, labels)
                i+=1
                if i>=10:
                    break
        # return train_loss/len(self.train)
        return train_loss/i

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
        # micro_f1 = f1_score(label_ts.cpu(), predict_ts.cpu(), average='micro')
        # print('Accuracy, Macro F1, Micro F1 of Device %d on the %d test cases: %.2f %%, %.2f, %.2f' % (self.id, len(label_ts), acc*100, macro_f1, micro_f1))
        print('Accuracy, Macro F1 of Device %d on the %d test cases: %.2f %%, %.2f %%' % (self.id, len(label_ts), acc*100, macro_f1*100))
        self.test_metrics.append([acc, macro_f1])


    def local_ensemble_test(self, model_list, voting = 'soft'):
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
                    predicted = torch.tensor([mode([out_hard[i][j] for i in range(len(out_hard))]) for j in range(len(out_hard[0]))]).to(self.device)

                predict_ts = torch.cat([predict_ts, predicted], 0)
                label_ts = torch.cat([label_ts, labels], 0)
        acc = accuracy_score(label_ts.cpu(), predict_ts.cpu())
        macro_f1 = f1_score(label_ts.cpu(), predict_ts.cpu(), average='macro')
        # micro_f1 = f1_score(label_ts.cpu(), predict_ts.cpu(), average='micro')
        # print('Accuracy, Macro F1, Micro F1 of Device %d on the %d test cases: %.2f %%, %.2f, %.2f' % (self.id, len(label_ts), acc*100, macro_f1, micro_f1))
        print('Accuracy, Macro F1 of Device %d on the %d test cases: %.2f %%, %.2f %%' % (self.id, len(label_ts), acc*100, macro_f1*100))
        self.test_metrics.append([acc, macro_f1])

def get_csd_loss(model, mu, omega):
    loss_set = []
    for name, param in model.named_parameters():
        theta = model.state_dict()[name]
        # omega_dropout = torch.rand(omega[name].size()).cuda() if cuda else torch.rand(omega[name].size())
        # omega_dropout[omega_dropout>0.5] = 1.0
        # omega_dropout[omega_dropout <= 0.5] = 0.0

        loss_set.append((0.5) * (omega[name] * ((theta - mu[name]) ** 2)).sum())

    return sum(loss_set)