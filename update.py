#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import utils
import copy
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, testset, idxs, test_idxs):
        self.args = args
        self.trainloader, self.testloader = self.train_test(
            dataset, testset, list(idxs), list(test_idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_test(self, dataset, testset, idxs, test_idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        # for 5 sample, (4,1,1)<-test = val
        idxs_train = idxs[:]
        idxs_test = test_idxs[:]
        
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        
        testloader = DataLoader(DatasetSplit(testset, idxs_test),
                                batch_size=self.args.local_bs, shuffle=False)
        return trainloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        
        target_model = copy.deepcopy(model)

        if self.args.diff_priv == 1:
            target_params_variables = dict()
            for name, param in target_model.named_parameters():
                target_params_variables[name] = target_model.state_dict()[name].clone().detach().requires_grad_(False)
        
        

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()

                output = model(images)
                criteria = nn.CrossEntropyLoss()
                loss = criteria(output, labels)
                loss.backward()
                optimizer.step()

                if self.args.diff_priv == 1:
                    #print('adding diff privacy')
                    model_norm = utils.model_dist_norm(model, target_params_variables)
                    if model_norm > self.args.s_norm:
                        norm_scale = self.args.s_norm / (model_norm)
                        for name, layer in model.named_parameters():
                            clipped_difference = norm_scale * (layer.data - target_model.state_dict()[name])
                            layer.data.copy_(target_model.state_dict()[name] + clipped_difference)

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):

        with torch.no_grad():
            model.eval()
            loss, total, correct = 0.0, 0.0, 0.0

            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Inference
                outputs = model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        accuracy = correct/total
        return accuracy, loss

