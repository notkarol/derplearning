import torch.nn as nn
from derp.models.blocks import ConvBlock, LinearBlock, ViewBlock

class Pyramid(nn.Module):

    def __init__(self, in_dim, out_dim, verbose=True):
        super(Pyramid, self).__init__()
        dim = in_dim.copy()
        self.c1 = ConvBlock(dim, 32, 5, stride=2, verbose=verbose)
        self.c2a = ConvBlock(dim, 80, 3, verbose=verbose)
        self.c2b = ConvBlock(dim, 40, 3, pool='max', verbose=verbose)
        self.c3a = ConvBlock(dim, 96, 3, verbose=verbose)
        self.c3b = ConvBlock(dim, 48, 3, pool='max', verbose=verbose)
        self.c4a = ConvBlock(dim, 112, 3, verbose=verbose)
        self.c4b = ConvBlock(dim, 56, 3, pool='max', verbose=verbose)
        self.c5a = ConvBlock(dim, 128, 3, verbose=verbose)
        self.c5b = ConvBlock(dim, 64, 3, pool='max', verbose=verbose)
        self.c6a = ConvBlock(dim, 144, 3, verbose=verbose)
        self.c6b = ConvBlock(dim, 72, int(dim[-2]), padding=0, verbose=verbose)
        self.view = ViewBlock(dim, verbose=verbose)
        self.fc1 = LinearBlock(dim, 64, verbose=verbose)
        self.fc2 = LinearBlock(dim, out_dim, activation=False, verbose=verbose)

    def forward(self, x):
        out = self.c1(x)
        out = self.c2a(out)
        out = self.c2b(out)
        out = self.c3a(out)
        out = self.c3b(out)
        out = self.c4a(out)
        out = self.c4b(out)
        out = self.c5a(out)
        out = self.c5b(out)
        out = self.c6a(out)
        out = self.c6b(out)
        out = self.view(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
