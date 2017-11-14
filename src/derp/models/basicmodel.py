import torch.nn as nn
from derp.models.blocks import ConvBlock, LinearBlock, ViewBlock
 
class BasicModel(nn.Module):

    def __init__(self, in_dim, out_dim, verbose=True):
        super(BasicModel, self).__init__()
        dim = in_dim.copy()
        self.c1 = ConvBlock(dim, 32, 5, stride=2, verbose=verbose)
        self.c2 = ConvBlock(dim, 40, 5, stride=2, verbose=verbose)
        self.c3 = ConvBlock(dim, 48, 5, stride=2, verbose=verbose)
        self.c4 = ConvBlock(dim, 56, 3, pool='max', verbose=verbose)
        self.c5 = ConvBlock(dim, 64, 3, pool='max', verbose=verbose)
        self.view = ViewBlock(dim, verbose=verbose)
        self.fc1 = LinearBlock(dim, 32, dropout=0.2, verbose=verbose)
        self.fc2 = LinearBlock(dim, out_dim, activation=False, verbose=verbose)

    def forward(self, x):
        out = self.c1(x)
        out = self.c2(out)
        out = self.c3(out)
        out = self.c4(out)
        out = self.c5(out)
        out = self.view(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
