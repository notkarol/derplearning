
import torch.nn as nn
from derp.models.blocks import *

class Resnet17Model(nn.Module):
    def __init__(self, in_dim, out_dim, verbose=True):
        super(Resnet17Model, self).__init__()
        dim = in_dim.copy()
        self.c1 = ConvBlock(dim, 32, kernel_size=5, stride=2, verbose=verbose)
        self.c2a = ResnetBlock(dim, 32, verbose=verbose)
        self.c2b = ResnetBlock(dim, 32, pool='max', verbose=verbose)
        self.c3a = ResnetBlock(dim, 48, verbose=verbose)
        self.c3b = ResnetBlock(dim, 48, pool='max', verbose=verbose)
        self.c4a = ResnetBlock(dim, 64, verbose=verbose)
        self.c4b = ResnetBlock(dim, 64, pool='max', verbose=verbose)
        self.c5 = ConvBlock(dim, 64, kernel_size=3, padding=0, verbose=verbose)
        self.pool = PoolBlock(dim, 'avg', verbose=verbose)
        self.view = ViewBlock(dim, verbose=verbose)
        self.fc1 = LinearBlock(dim, out_dim, activation=False, verbose=verbose)

    def forward(self, x):
        out = self.c1(x)
        out = self.c2a(out)
        out = self.c2b(out)
        out = self.c3a(out)
        out = self.c3b(out)
        out = self.c4a(out)
        out = self.c4b(out)
        out = self.c5(out)
        out = self.pool(out)
        out = self.view(out)
        out = self.fc1(out)
        return out
    
