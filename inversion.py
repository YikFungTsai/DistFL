from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.transforms as transforms
import scipy.stats
from torchvision import models
import numpy
import numpy as np
import os
import glob
import collections
import torchvision

from models import ModifiedVGG11Model

class DeepInversionFeatureHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(var.type()) - mean, 2)
        self.r_feature = r_feature
        self.input = input[0]
        self.r_var = module.running_var
        self.r_mean = module.running_mean
        self.weight = module.weight

    def close(self):
        self.hook.remove()

def get_images(net, local_weight, args, bs, map_size=0,epochs=1000, idx=-1, var_scale=0.00005,
               competitive_scale=0.01, global_iteration=None,
               optimizer = None, inputs = None, bn_reg_scale = 0.0, random_labels = False, l2_coeff=0.0):

    kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    best_cost = 1e6

    inputs.data = torch.randn((bs, 3, 32, 32), requires_grad=True, device='cuda')
  
    criterion = nn.CrossEntropyLoss()

    optimizer.state = collections.defaultdict(dict)  

    if random_labels:
        targets = torch.LongTensor([random.randint(0,9) for _ in range(bs)]).to('cuda')
    else:
        targets = torch.LongTensor([0]*(bs)).to('cuda')
       
    loss_r_feature_layers = []
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))
    
    lim_0, lim_1 = 2, 2

    for epoch in range(epochs):
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(inputs, shifts=(off1,off2), dims=(2,3))
        optimizer.zero_grad()
        net.zero_grad()
        outputs = net(inputs_jit)
        loss = criterion(outputs, targets)
        loss_target = loss.item()
        diff1 = inputs_jit[:,:,:,:-1] - inputs_jit[:,:,:,1:]
        diff2 = inputs_jit[:,:,:-1,:] - inputs_jit[:,:,1:,:]
        diff3 = inputs_jit[:,:,1:,:-1] - inputs_jit[:,:,:-1,1:]
        diff4 = inputs_jit[:,:,:-1,:-1] - inputs_jit[:,:,1:,1:]
        loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
        loss = loss + var_scale*loss_var

        r_mean = []
        r_var = []
        r_weight = []
        r_inputs = []
        for mod in loss_r_feature_layers:
            r_mean.append(mod.r_mean.cpu().detach().numpy())
            r_var.append(mod.r_var.cpu().detach().numpy())
            r_weight.append(mod.weight.cpu().detach().numpy())
            r_inputs.append(mod.input.cpu().detach().numpy())
            
        sort_weight = np.sort(np.concatenate(r_weight))
        threshold = sort_weight[int(sort_weight.shape[0]*0.5)]
        
        loss_distr=0
        for i in range(len(r_weight)):
            del_list = []
            sort_weight = np.sort(r_weight[i])
            threshold = sort_weight[int(sort_weight.shape[0]*0.5)]
            for j in range(len(r_weight[i])):
                if r_weight[i][j] < threshold:
                    del_list.append(j)

            r_weight[i]=numpy.delete(r_weight[i],del_list)
            r_mean[i]=numpy.delete(r_mean[i],del_list)
            r_var[i]=numpy.delete(r_var[i],del_list)
            r_inputs[i]=numpy.delete(r_inputs[i],del_list,axis=1)
            nch = torch.from_numpy(r_inputs[i]).shape[1]
            mean = torch.from_numpy(r_inputs[i]).mean([0, 2, 3])
            var = torch.from_numpy(r_inputs[i]).permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
            loss_distr += torch.norm(torch.from_numpy(r_var[i]).data.type(var.type()) - var, 2) + torch.norm(torch.from_numpy(r_mean[i]).data.type(var.type()) - mean, 2)

        loss_distr.requires_grad_(True)
        loss_distr.to('cuda')

        loss = bn_reg_scale * loss_distr

        if best_cost > loss:
            best_cost = loss
            best_inputs = inputs.data

        loss.backward()

        optimizer.step()

    map_size = map_size
    r = [[] for i in range(map_size)]
    p = [[0]*map_size for i in range(map_size)]
    for i in range(map_size):
        net_student = torchvision.models.resnet50(pretrained=False)
        net_student.fc = nn.Linear(net_student.fc.in_features, out_features=65)
        net_student.load_state_dict(local_weight[i])
        net_student.to('cuda')
        net_student.eval()
        outputs_student=net_student(best_inputs)
        r[i] = F.softmax(outputs_student).cpu().detach().numpy()
        _, predicted_std = outputs_student.max(1)
        net_student.train()
    for i in range(map_size):
        for j in range(map_size):
            x = r[i].flatten()
            y = r[j].flatten()
            p[i][j] = scipy.stats.entropy(x, y)
    print(p)

    return p

def run_inversion(inversion_net, local_weight, args, map_size):
    #net = ModifiedVGG11Model(args=args)
    net = torchvision.models.resnet50(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, out_features=65)
    net.load_state_dict(inversion_net)
    net.to('cuda')
    criterion = nn.CrossEntropyLoss()
    bs = 200
    inputs = torch.randn((bs, 3, 224,224), requires_grad=True, device='cuda', dtype=torch.float)
    optimizer_di = optim.Adam([inputs], lr=0.1)

    net.eval()
    
    cudnn.benchmark = True
    batch_idx = 0
    global_iteration = 0
    print("Starting model inversion")

    result = get_images(net=net, local_weight=local_weight, args=args, bs=bs, map_size=map_size, epochs=2000, idx=0,
                        competitive_scale=0.0,
                        global_iteration=global_iteration, 
                        optimizer=optimizer_di, inputs=inputs, bn_reg_scale=0.1,
                        var_scale=2.5e-5, random_labels=False, l2_coeff=3e-8)
    return result
