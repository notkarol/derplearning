
import torch.nn as nn
from derp.models.blocks import *

class Resnet33Model(nn.Module):
    def __init__(self, in_dim, n_status, n_out, verbose=True):
        super(Resnet33Model, self).__init__()
        dim = in_dim.copy()
        self.c1 = ConvBlock(dim, 32, kernel_size=5, stride=2, verbose=verbose)
        self.c2a = ResnetBlock(dim, 32, verbose=verbose)
        self.c2b = ResnetBlock(dim, 32, verbose=verbose)
        self.c2c = ResnetBlock(dim, 32, pool='max', verbose=verbose)
        self.c3a = ResnetBlock(dim, 32, verbose=verbose)
        self.c3b = ResnetBlock(dim, 32, verbose=verbose)
        self.c3c = ResnetBlock(dim, 32, verbose=verbose)
        self.c3d = ResnetBlock(dim, 32, verbose=verbose)
        self.c3e = ResnetBlock(dim, 32, verbose=verbose)
        self.c3f = ResnetBlock(dim, 32, pool='max', verbose=verbose)
        self.c4a = ResnetBlock(dim, 32, verbose=verbose)
        self.c4b = ResnetBlock(dim, 32, verbose=verbose)
        self.c4c = ResnetBlock(dim, 32, verbose=verbose)
        self.c4d = ResnetBlock(dim, 32, verbose=verbose)
        self.c4e = ResnetBlock(dim, 32, verbose=verbose)
        self.c4f = ResnetBlock(dim, 32, pool='max', verbose=verbose)
        self.c5 = ConvBlock(dim, 64, kernel_size=3, padding=0, verbose=verbose)
        self.view = ViewBlock(dim, verbose=verbose)
        dim[0] += n_status
        self.fc1 = LinearBlock(dim, n_out, activation=False, verbose=verbose)

    def forward(self, x, status):
        out = self.c1(x)
        
        out = self.c2a(out)
        out = self.c2b(out)
        out = self.c2c(out)
        out = self.c3a(out)
        out = self.c3b(out)
        out = self.c3c(out)
        out = self.c3d(out)
        out = self.c3e(out)
        out = self.c3f(out)
        out = self.c4a(out)
        out = self.c4b(out)
        out = self.c4c(out)
        out = self.c4d(out)
        out = self.c4e(out)
        out = self.c4f(out)

        out = self.c5(out)
        out = self.view(out)
        out = torch.cat((out, status), 1)
        out = self.fc1(out)
        return out
    
