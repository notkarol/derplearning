import torch
import torch.nn as nn
from derp.models.blocks import ConvBlock, LinearBlock, ViewBlock

class PilotNet(nn.Module):

    def __init__(self, in_dim, n_status, n_out, verbose=True):
        super(PilotNet, self).__init__()
        dim = in_dim.copy()
        self.c1 = ConvBlock(dim, 24, 5, stride=2, padding=0, verbose=verbose)
        self.c2 = ConvBlock(dim, 36, 5, stride=2, padding=0, verbose=verbose)
        self.c3 = ConvBlock(dim, 48, 5, stride=2, padding=0, verbose=verbose)
        self.c4 = ConvBlock(dim, 64, 3, stride=1, padding=0, verbose=verbose)
        self.c5 = ConvBlock(dim, 64, 3, stride=1, padding=0, verbose=verbose)
        self.view = ViewBlock(dim, verbose=verbose)
        self.fc1 = LinearBlock(dim, 100, verbose=verbose)
        dim[0] += n_status
        self.fc2 = LinearBlock(dim, 50, verbose=verbose)
        self.fc3 = LinearBlock(dim, 10, verbose=verbose)
        self.fc4 = LinearBlock(dim, n_out, activation=False, verbose=verbose)

        
    def forward(self, x, status):
        out = self.c1(x)
        out = self.c2(out)
        out = self.c3(out)
        out = self.c4(out)
        out = self.c5(out)
        out = self.view(out)
        out = self.fc1(out)
        out = torch.cat((out, status), 1)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out
