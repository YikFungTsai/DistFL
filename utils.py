#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
import math

def get_dataset(args):

    test_dataset = []
    train_dataset = []
    for i in range(20):
        train_dir = '/_dataset/office_home/client_set/client' + str(i+1)
        test_dir = '/_dataset/office_home/'
        if i in range(0,5):
            test_dir += 'Art_test'
        elif i in range(5,10):
            test_dir += 'Clipart_test'
        elif i in range(10,15):
            test_dir += 'Product_test'
        elif i in range(15,20):
            test_dir += 'Real_World_test'
        train_transform = transforms.Compose(
            [transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        test_transform = transforms.Compose(
            [transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        train_dataset.append(datasets.ImageFolder(train_dir, transform=train_transform))

        test_dataset.append(datasets.ImageFolder(test_dir, transform=test_transform))

    return train_dataset, test_dataset

class UnionFind(object):
    def __init__(self, n):
        self.uf = [-1 for i in range(n)]    
        self.sets_count = n            

    def find(self, p):
        if self.uf[p] < 0:
            return p
        self.uf[p] = self.find(self.uf[p])
        return self.uf[p]

    def union(self, p, q):
        proot = self.find(p)
        qroot = self.find(q)
        if proot == qroot:
            return
        elif self.uf[proot] > self.uf[qroot]: 
            self.uf[qroot] += self.uf[proot]
            self.uf[proot] = qroot
        else:
            self.uf[proot] += self.uf[qroot]  
            self.uf[qroot] = proot
        self.sets_count -= 1           

    def is_connected(self, p, q):
        return self.find(p) == self.find(q)   

def classified_group(a,n):
    union_class = [0]*(n)
    for i in range(n):
        union_class[i]= a.find(i)
    
    model_to_client = [[] for i in range(len(set(union_class)))]
    
    for index,data in enumerate(set(union_class)):
        for j in range(n):
            if union_class[j] == data:
                model_to_client[index].append(j)
    
    client_to_model = [0]*n
    
    for i in range(len(model_to_client)):
        for j in model_to_client[i]:
            client_to_model[j] = i
            
    return model_to_client, client_to_model

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def grouped_average_weights(w, model_to_client, grouped_global_model):
    """
    Returns the average of the weights in groups.
    """
    w_avg = []
    print(model_to_client)
    for model in range(len(model_to_client)):
        if len(model_to_client[model]) > 0:
            print(len(model_to_client[model]))
            w_avg.append(copy.deepcopy(w[model_to_client[model][0]]))
            print(len(w_avg))
            print(model)
            print(w_avg[model].keys())
            for key in w_avg[model].keys():
                if len(model_to_client[model]) > 1:
                    for i in (model_to_client[model][1:]):
                        w_avg[model][key] += w[i][key]
                w_avg[model][key] = torch.div(w_avg[model][key], len(model_to_client[model]))
        else:
            w_avg.append(copy.deepcopy(grouped_global_model[model].state_dict()))


    return w_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def model_dist_norm(model, target_params):
    squared_sum = 0
    for name, layer in model.named_parameters():
        if 'running_' in name or '_tracked' in name:
            continue
        squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
    return math.sqrt(squared_sum)