import torch
import torch.nn as nn
from derp.models.blocks import ConvBlock, LinearBlock, ViewBlock
 
class AModel(nn.Module):

    def __init__(self, in_dim, n_status, n_out, verbose=True):
        super(AModel, self).__init__()
        self.n_status = n_status
        dim = in_dim.copy()
        self.c1a = ConvBlock(dim, 12, 5, stride=2, padding=2,
                             batchnorm=False, verbose=verbose)
        self.c2a = ConvBlock(dim, 16, 3, pool='max', padding=1,
                             batchnorm=False, verbose=verbose)
        self.c3a = ConvBlock(dim, 20, 3, pool='max', padding=1,
                             batchnorm=False, verbose=verbose)
        self.c4a = ConvBlock(dim, 24, 3, pool='max', padding=1,
                             batchnorm=False, verbose=verbose)
        self.c5a = ConvBlock(dim, 28, 3, pool='max', padding=1,
                             batchnorm=False, verbose=verbose)
        self.c6a = ConvBlock(dim, 32, 2, padding=0,
                             batchnorm=False, verbose=verbose)
        self.view = ViewBlock(dim, verbose=verbose)
        dim[0] += n_status
        self.fc1 = LinearBlock(dim, 32, verbose=verbose)
        self.fc2 = LinearBlock(dim, 32, verbose=verbose)
        self.fc3 = LinearBlock(dim, n_out, activation=False, verbose=verbose)

    def forward(self, x, status):
        out = self.c1a(x)
        out = self.c2a(out)
        out = self.c3a(out)
        out = self.c4a(out)
        out = self.c5a(out)
        out = self.c6a(out)
        out = self.view(out)
        if self.n_status:
            out = torch.cat((out, status), 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
