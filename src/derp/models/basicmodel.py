import torch
import torch.nn as nn
from derp.models.blocks import ConvBlock, LinearBlock, ViewBlock
 
class BasicModel(nn.Module):

    def __init__(self, in_dim, n_status, n_out, verbose=True):
        super(BasicModel, self).__init__()
        dim = in_dim.copy()
        self.c1 = ConvBlock(dim, 24, 5, stride=2, verbose=verbose)
        self.c2 = ConvBlock(dim, 24, 3, pool='max', verbose=verbose)
        self.c3 = ConvBlock(dim, 32, 3, pool='max', verbose=verbose)
        self.c4 = ConvBlock(dim, 40, 3, pool='max', verbose=verbose)
        self.c5 = ConvBlock(dim, 48, 3, pool='max', verbose=verbose)
        self.view = ViewBlock(dim, verbose=verbose)
        dim[0] += n_status
        self.fc1 = LinearBlock(dim, 32, verbose=verbose)
        self.fc2 = LinearBlock(dim, n_out, activation=False, verbose=verbose)

    def forward(self, x, status):
        out = self.c1(x)
        out = self.c2(out)
        out = self.c3(out)
        out = self.c4(out)
        out = self.c5(out)
        out = self.view(out)
        out = torch.cat((out, status), 1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
