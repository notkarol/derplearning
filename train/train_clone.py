#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import derputil
from derpfetcher import DerpFetcher

class Block(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, stride=1,
                 padding=None, pool=None):
        super(Block, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)

        if pool == 'max':
            self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        elif pool == 'avg':
            self.pool = nn.AvgPool2d(2, stride=2, padding=0)
        else:
            self.pool = None

        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.pool is not None:
            out = self.pool(out)
        return out

    
class ModelA(nn.Module):
    def __init__(self, config):
        super(ModelA, self).__init__()
        self.lrn = nn.CrossMapLRN2d(3)
        self.layer1 = Block(3,  96, 5, stride=2)
        self.layer2 = Block(96, 64, 3, pool='max')
        self.layer3 = Block(64, 64, 3, pool='max')
        self.layer4 = Block(64, 64, 3, pool='max')
        self.fc1 = nn.Linear(8 * 2 * 64, 64)
        self.fc2 = nn.Linear(64, len(config['predict']))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.lrn(x)
        out = self.layer1(out)
        out = nn.functional.dropout2d(out, p=0.2, training=self.training)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = nn.functional.dropout(out, p=0.5, training=self.training)
        out = self.fc2(out)
        return out

    
def step(epoch, config, model, loader, optimizer, criterion, is_train):
    if is_train:
        model.train()
    else:
        model.eval()
    step_loss = []
    for batch_idx, (example, state) in enumerate(loader):
        label = torch.stack([state[x] for x in config['predict']], dim=1).float()
        example, label = Variable(example.cuda()), Variable(label.cuda())
        if is_train:
            optimizer.zero_grad()
        out = model(example)
        loss = criterion(out, label)
        step_loss.append(loss.data.mean())
        if is_train:
            loss.backward()
            optimizer.step()
    return np.mean(step_loss), np.std(step_loss)


def main():

    # Load arguemnts
    config = derputil.loadConfig(sys.argv[1])
    if len(sys.argv) >= 3:
        gpu = sys.argv[2]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # Make sure we have somewhere to run the experiment
    experiment_path = os.path.join(os.environ["DERP_SCRATCH"], config['name'])

    # prepare data fetchers
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                std=[0.2, 0.2, 0.2])])
    train_dir = os.path.join(experiment_path, 'train')
    eval_dir = os.path.join(experiment_path, 'eval')
    train_set = DerpFetcher(train_dir, transform)
    eval_set = DerpFetcher(eval_dir, transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'],
                                               shuffle='True', num_workers=config['num_threads'])
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=config['batch_size'],
                                              num_workers=config['num_threads'])

    model = ModelA(config).cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), config['learning_rate'])

    lowest_loss = 1
    for epoch in range(config['num_epochs']):
        tmean, tstd = step(epoch, config, model, train_loader, optimizer, criterion, is_train=True)
        emean, estd = step(epoch, config, model, eval_loader, optimizer, criterion, is_train=False)
        print("epoch %03i train loss:%.6f std:%.6f eval loss:%.6f std:%.6f]" %
              (epoch, tmean, tstd, emean, estd))
        if emean < lowest_loss:
            lowest_loss = emean
            torch.save(model, os.path.join(experiment_path, "model_%03i.pt" % epoch))
    
if __name__ == "__main__":
    main()
