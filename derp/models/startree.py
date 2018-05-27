import torch
import torch.nn as nn
from derp.models.blocks import ConvBlock, LinearBlock, ViewBlock, PoolBlock

class StarTree(nn.Module):

    def __init__(self, in_dim, n_status, n_out, verbose=True):
        super(StarTree, self).__init__()
        self.n_status = n_status
        dim = in_dim.copy()
        self.c1 = ConvBlock(dim, 64, 5, stride=2, verbose=verbose)
        self.c2 = ConvBlock(dim, 16, 3, verbose=verbose)
        self.c3 = ConvBlock(dim, 32, 3, pool='max', verbose=verbose)
        self.c4 = ConvBlock(dim, 24, 3, verbose=verbose)
        self.c5 = ConvBlock(dim, 48, 3, pool='max', verbose=verbose)
        self.c6 = ConvBlock(dim, 32, 3, verbose=verbose)
        self.c7 = ConvBlock(dim, 64, 3, pool='max', verbose=verbose)
        self.c8 = ConvBlock(dim, 40, 3, verbose=verbose)
        self.c9 = ConvBlock(dim, 80, 2, pool='max', verbose=verbose, dropout=0.25)
        self.view = ViewBlock(dim, verbose=verbose)
        dim[0] += n_status
        self.fc1 = LinearBlock(dim, 50, verbose=verbose)
        self.fc2 = LinearBlock(dim, n_out, activation=False, verbose=verbose)

        self.n_params = sum([self._modules[x].n_params for x in self._modules])
        if verbose:
            print("StarTree                                  params %9i" % self.n_params)
        
    def forward(self, x, status):
        out = self.c1(x)
        out = self.c2(out)
        out = self.c3(out)
        out = self.c4(out)
        out = self.c5(out)
        out = self.c6(out)
        out = self.c7(out)
        out = self.c8(out)
        out = self.c9(out)
        out = self.view(out)
        if self.n_status:
            out = torch.cat((out, status), 1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
