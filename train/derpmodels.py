import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, stride=1,
                 padding=None, pool=None):
        super(Block, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_features)
        self.elu = nn.ELU(inplace=True)

        if pool == 'max':
            self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        elif pool == 'avg':
            self.pool = nn.AvgPool2d(2, stride=2, padding=0)
        else:
            self.pool = None
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        if self.pool is not None:
            out = self.pool(out)
        return out

    
class ModelA(nn.Module):
    def __init__(self, config):
        super(ModelA, self).__init__()
        self.c1 = Block(config['patch']['depth'],  64, 5, stride=2)
        self.c2 = Block(64, 64, 3, pool='max')
        self.c3 = Block(64, 64, 3, pool='max')
        self.c4 = Block(64, 64, 3, pool='max')
        self.fc1 = nn.Linear(8 * 2 * 64, 64)
        self.fc2 = nn.Linear(64, len(config['states']))
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.c1(x)
        out = self.c2(out)
        out = self.c3(out)
        out = self.c4(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.elu(out)
        out = nn.functional.dropout(out, p=0.5, training=self.training)
        out = self.fc2(out)
        return out

    
class ModelB(nn.Module):
    def __init__(self, config):
        super(ModelB, self).__init__()
        self.c1 = Block(config['patch']['depth'],  64, 5, stride=2)
        self.c2a = Block(64, 64, 3)
        self.c2b = Block(64, 64, 3, pool='max')
        self.c3a = Block(64, 64, 3)
        self.c3b = Block(64, 64, 3, pool='max')
        self.c4 = Block(64, 64, 3, pool='max')
        self.fc1 = nn.Linear(8 * 2 * 64, 64)
        self.fc2 = nn.Linear(64, len(config['states']))
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.c1(x)
        out = self.c2a(out)
        out = self.c2b(out)
        out = self.c3a(out)
        out = self.c3b(out)
        out = self.c4(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.elu(out)
        out = nn.functional.dropout(out, p=0.5, training=self.training)
        out = self.fc2(out)
        return out
    
