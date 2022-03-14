#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import models
class ModifiedVGG11Model(nn.Module):
    def __init__(self, args):
        super(ModifiedVGG11Model, self).__init__()

        model = models.vgg11_bn(pretrained=False)
        self.features = model.features

        for param in self.features.parameters():
            param.requires_grad = True

        self.avgpool = nn.AvgPool2d(1, stride=1)
        classifier = nn.Sequential(
            nn.Linear(512, args.num_classes),
        )
        self.fc = classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

