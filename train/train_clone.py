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
            self.pool = nn.MaxPool2d(2, stride=2, padding=0, return_indices=True)
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
        self.layer1 = Block( 3, 64, 7, pool='max')
        self.layer2 = Block(64, 64, 5, pool='max')
        self.layer3 = Block(64, 64, 3, pool='max')
        self.layer4 = Block(64, 64, 3, pool='max')
        self.fc1 = nn.Linear(8 * 2 * 64, 64)
        self.fc2 = nn.Linear(64, len(config['predict']))
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = nn.functional.dropout(out, training=self.training)
        out = self.fc2(out)
        print(out)
        return out

    
def train_model(epoch, model, loader, optimizer, criterion):
    model.train()
    train_loss = 0
    for batch_idx, (example, label) in enumerate(loader):
        example, label = Variable(example.cuda()), Variable(label.cuda())
        optimizer.zero_grad()
        out = model(example)
        loss = criterion(out, label)
        train_loss += loss
        loss.backward()
        optimizer.step()
        print(batch_idx, loss)
    print("Train [%03i]: %.6f" % (epoch, train_loss))

        
def eval_model(epoch, model, loader, optimizer):
    model.eval()
    eval_loss = 0
    for batch_idx, (example, label) in enumerate(loader):
        example, label = Variable(example.cuda()), Variable(label.cuda())
        out = model(example)
        eval_loss += criterion(out, label)
    print("Eval  [%03i]: %.6f" % (epoch, eval_loss))
        
    
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
    train_set = datasets.ImageFolder(train_dir, transform)
    eval_set = datasets.ImageFolder(eval_dir, transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'],
                                               shuffle='True', num_workers=config['num_threads'])
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=config['batch_size'],
                                              num_workers=config['num_threads'])

    model = ModelA(config).cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), config['learning_rate'])

    for epoch in range(config['num_epochs']):
        train_model(epoch, model, train_loader, optimizer, criterion)
        eval_model(epoch, model, eval_loader, optimizer)
    

if __name__ == "__main__":
    main()
