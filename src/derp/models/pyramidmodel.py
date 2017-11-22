import torch.nn as nn
from derp.models.blocks import ConvBlock, LinearBlock, ViewBlock

class PyramidModel(nn.Module):

    def __init__(self, in_dim, n_status, n_out, verbose=True):
        super(PyramidModel, self).__init__()
        dim = in_dim.copy()
        self.c1 = ConvBlock(dim, 24, 5, stride=2, verbose=verbose)
        self.c2 = ConvBlock(dim, 32, 3, pool='max', verbose=verbose)
        self.c3a = ConvBlock(dim, 40, 3, verbose=verbose)
        self.c3b = ConvBlock(dim, 48, 3, pool='max', verbose=verbose)
        self.c4a = ConvBlock(dim, 56, 3, verbose=verbose)
        self.c4b = ConvBlock(dim, 64, 3, pool='max', verbose=verbose)
        self.c5a = ConvBlock(dim, 72, 3, verbose=verbose)
        self.c5b = ConvBlock(dim, 80, 3, pool='max', verbose=verbose)
        self.view = ViewBlock(dim, verbose=verbose)
        dim[0] += n_status
        self.fc1 = LinearBlock(dim, 32, verbose=verbose)
        self.fc2 = LinearBlock(dim, n_out, activation=False, verbose=verbose)

    def forward(self, x, status):
        out = self.c1(x)
        out = self.c2(out)
        out = self.c3a(out)
        out = self.c3b(out)
        out = self.c4a(out)
        out = self.c4b(out)
        out = self.c5a(out)
        out = self.c5b(out)
        out = self.view(out)
        out = torch.cat((out, status), 1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
