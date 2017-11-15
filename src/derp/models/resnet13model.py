
import torch.nn as nn
from derp.models.blocks import *

class Resnet13Model(nn.Module):
    def __init__(self, in_dim, out_dim, verbose=True):
        super(Resnet13Model, self).__init__()
        dim = in_dim.copy()
        self.c1 = ConvBlock(dim, 32, kernel_size=5, stride=2, verbose=verbose)
        self.c2 = ResnetBlock(dim, 32, pool='max', verbose=verbose)
        self.c3 = ResnetBlock(dim, 32, pool='max', verbose=verbose)
        self.c4 = ResnetBlock(dim, 32, pool='max', verbose=verbose)
        self.c5 = ConvBlock(dim, 64, kernel_size=3, padding=0, verbose=verbose)
        self.view = ViewBlock(dim, verbose=verbose)
        self.fc1 = LinearBlock(dim, out_dim, activation=False, verbose=verbose)

    def forward(self, x):
        out = self.c1(x)
        out = self.c2(out)
        out = self.c3(out)
        out = self.c4(out)
        out = self.c5(out)
        out = self.view(out)
        out = self.fc1(out)
        return out
    
