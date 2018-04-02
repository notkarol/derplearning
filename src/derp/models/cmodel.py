import torch
import torch.nn as nn
from derp.models.blocks import *

class CModel(nn.Module):
    def __init__(self, in_dim, n_status, n_out, verbose=True):
        super(CModel, self).__init__()
        self.n_status = n_status
        dim = in_dim.copy()
        self.c1 = ConvBlock(dim, 96, kernel_size=5, stride=2, verbose=verbose)
        self.c2a = ResnetBlock(dim, 32, verbose=verbose)
        self.c2b = ResnetBlock(dim, 32, verbose=verbose)
        self.c2c = ResnetBlock(dim, 48, pool='avg', verbose=verbose)
        self.c3a = ResnetBlock(dim, 48, verbose=verbose)
        self.c3b = ResnetBlock(dim, 48, verbose=verbose)
        self.c3c = ResnetBlock(dim, 48, verbose=verbose)
        self.c3d = ResnetBlock(dim, 48, verbose=verbose)
        self.c3e = ResnetBlock(dim, 48, verbose=verbose)
        self.c3f = ResnetBlock(dim, 64, pool='avg', verbose=verbose)
        self.c4a = ResnetBlock(dim, 64, verbose=verbose)
        self.c4b = ResnetBlock(dim, 64, verbose=verbose)
        self.c4c = ResnetBlock(dim, 64, verbose=verbose)
        self.c4d = ResnetBlock(dim, 64, verbose=verbose)
        self.c4e = ResnetBlock(dim, 64, verbose=verbose)
        self.c4f = ResnetBlock(dim, 96, pool='max', verbose=verbose)
        self.view = ViewBlock(dim, verbose=verbose)
        dim[0] += n_status
        self.fc1 = LinearBlock(dim, 128, verbose=verbose)
        self.fc2 = LinearBlock(dim, 128, verbose=verbose)
        self.fc3 = LinearBlock(dim, n_out, activation=False, verbose=verbose)

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
        out = self.view(out)
        if self.n_status:
            out = torch.cat((out, status), 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    
