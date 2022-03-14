#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np

import torch
import torch.nn as nn

from options import args_parser
from update import LocalUpdate
from models import ModifiedVGG11Model
from utils import get_dataset, average_weights, grouped_average_weights, exp_details, UnionFind, classified_group
from inversion import run_inversion
import torchvision

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(1)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'cifar':
            #global_model = ModifiedVGG11Model(args=args)
            global_model = torchvision.models.resnet18(pretrained=False)
            global_model.fc = nn.Linear(global_model.fc.in_features, out_features=10)
        elif args.dataset == 'oh':
            global_model = torchvision.models.resnet50(pretrained=True)
            global_model.fc = nn.Linear(global_model.fc.in_features, out_features=65)


    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    best_accuracy = 0.0
    val_loss_pre, counter = 0, 0

    for epoch in range(5):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        global_model.train()
        idxs_users = np.array([i for i in range(args.num_users)])
        for idx in idxs_users:
            args.local_ep = 5
            print(f'| Client : {idx} |')
            local_model = LocalUpdate(args=args, dataset=train_dataset[idx], testset=test_dataset[idx], 
                                      idxs=np.asarray(list(range(len(train_dataset[idx])))), test_idxs=np.asarray(list(range(len(test_dataset[idx])))))
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        global_model.eval()
        list_acc, list_loss = [], []
        for c in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset[c], testset=test_dataset[c],
                                      idxs=np.asarray(list(range(len(train_dataset[c])))), test_idxs=np.asarray(list(range(len(test_dataset[c])))))
            acc, loss = local_model.inference(model=copy.deepcopy(global_model))
            list_acc.append(acc)
            list_loss.append(loss)
            print('Client', c, 'Acc', acc)
        
        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

    slt = run_inversion(local_weights[0], local_weights, args=args, map_size=args.num_users)
    
    union = UnionFind(args.num_users)
    
    for i in range(args.num_users):
        for j in range(args.num_users):
            if slt[i][j] < 0.2:
               union.union(i, j)

    model_to_client, client_to_model = classified_group(union, args.num_users)
 
    print(model_to_client)
    print(client_to_model)
    grouped_global_model = []

    for i in range(np.array(model_to_client).shape[0]):
        grouped_global_model.append(copy.deepcopy(global_model))
    train_loss, train_accuracy = [], []
    for epoch in range(5, args.epochs):
        local_weights, local_losses = [], []
        print(f'\n | Grouped Global Training Round : {epoch+1} |\n')
        for i in range(np.array(model_to_client).shape[0]):
            grouped_global_model[i].train()
        idxs_users = np.array([i for i in range(args.num_users)])
        # training part
        for idx in idxs_users:
            print(f'| Client : {idx} |')
            local_model = LocalUpdate(args=args, dataset=train_dataset[idx], testset=test_dataset[idx], 
                                      idxs=np.asarray(list(range(len(train_dataset[idx])))), test_idxs=np.asarray(list(range(len(test_dataset[idx])))))
            w, loss = local_model.update_weights(model=copy.deepcopy(grouped_global_model[client_to_model[idx]]), global_round=epoch)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update grouped global weights
        grouped_global_weights = grouped_average_weights(local_weights, model_to_client, grouped_global_model)

        # update grouped global weights
        for i in range(len(grouped_global_model)):
            grouped_global_model[i].load_state_dict(grouped_global_weights[i])

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        
        for i in range(np.array(model_to_client).shape[0]):
            grouped_global_model[i].eval()

        for c in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset[c], testset=test_dataset[c],
                                      idxs=np.asarray(list(range(len(train_dataset[c])))), test_idxs=np.asarray(list(range(len(test_dataset[c])))))
            acc, loss = local_model.inference(model=grouped_global_model[client_to_model[c]])
            list_acc.append(acc)
            list_loss.append(loss)
            print('Client', c, 'Acc', acc)

        print('Average Test Accuracy: {:.2f}% \n'.format(100*(sum(list_acc)/len(list_acc))))
        train_accuracy.append(sum(list_acc)/len(list_acc))
        if train_accuracy[-1] > best_accuracy:
            best_accuracy = train_accuracy[-1]
