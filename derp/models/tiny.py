import torch
import torch.nn as nn
from derp.models.blocks import ConvBlock, LinearBlock, ViewBlock
 
class Tiny(nn.Module):

    def __init__(self, in_dim, n_status, n_out, verbose=True):
        super(Tiny, self).__init__()
        self.n_status = n_status
        dim = in_dim.copy()
        self.c1 = ConvBlock(dim, 32, 5, stride=2, batchnorm=False, verbose=verbose)
        self.c2 = ConvBlock(dim, 20, 3, pool='max', batchnorm=False, verbose=verbose)
        self.c3 = ConvBlock(dim, 24, 3, pool='max', batchnorm=False, verbose=verbose)
        self.c4 = ConvBlock(dim, 28, 3, pool='max', batchnorm=False, verbose=verbose)
        self.c5 = ConvBlock(dim, 32, 2, pool='max', batchnorm=False, verbose=verbose)
        self.view = ViewBlock(dim, verbose=verbose)
        dim[0] += n_status
        self.fc1 = LinearBlock(dim, n_out, activation=False, verbose=verbose)

        self.n_params = sum([self._modules[x].n_params for x in self._modules])
        if verbose:
            print("Tiny                                      params %9i" % self.n_params)

    def forward(self, x, status):
        out = self.c1(x)
        out = self.c2(out)
        out = self.c3(out)
        out = self.c4(out)
        out = self.c5(out)
        out = self.view(out)
        if self.n_status:
            out = torch.cat((out, status), 1)
        out = self.fc1(out)
        return out
