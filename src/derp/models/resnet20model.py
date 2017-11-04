
import torch.nn as nn
from derp.models.blocks import *

class Resnet20Model(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Resnet20Model, self).__init__()
        dim = in_dim.copy()
        self.c1 = ConvBlock(dim, 64, 5, stride=2) ; self.h /= 2 ; self.w /= 2
        self.c2a = ResnetBlock(dim, 64)
        self.c2b = ResnetBlock(dim, 64)
        self.c3a = ResnetBlock(dim, 64)
        self.c3b = ResnetBlock(dim, 64)
        self.c4a = ResnetBlock(dim, 64)
        self.c4b = ResnetBlock(dim, 64)
        self.avgpool = PoolBlock(nn.AvgPool2d((int(self.h), int(self.w))) ; self.w = 1 ; self.h = 1
        self.fc1 = nn.Linear(int(self.h * self.w * 64), len(config['fields']))

    def forward(self, x):
        out = self.c1(x)
        out = self.c2a(out)
        out = self.c2b(out)
        out = self.maxpool(out)
        out = self.c3a(out)
        out = self.c3b(out)
        out = self.maxpool(out)
        out = self.c4a(out)
        out = self.c4b(out)
        out = self.maxpool(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out
    
