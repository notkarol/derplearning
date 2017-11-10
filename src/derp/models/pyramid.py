import torch.nn as nn
from derp.models.blocks import ConvBlock, LinearBlock, ViewBlock

class Pyramid(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Pyramid, self).__init__()
        dim = in_dim.copy()
        self.c1 = ConvBlock(dim, 64, 5, stride=2)
        self.c2a = ConvBlock(dim, 32, 1)
        self.c2b = ConvBlock(dim, 96, 3, stride=2)
        self.c3a = ConvBlock(dim, 48, 1)
        self.c3b = ConvBlock(dim, 128, 3, stride=2)
        self.c4a = ConvBlock(dim, 64, 1)
        self.c4b = ConvBlock(dim, 160, 3, pool='max')
        self.c5a = ConvBlock(dim, 80, 1)
        self.c5b = ConvBlock(dim, 192, 3, pool='max')
        self.c6a = ConvBlock(dim, 96, 1)
        self.c6b = ConvBlock(dim, 224, int(dim[-2]), padding=0)
        self.view = ViewBlock(dim)
        self.fc1 = LinearBlock(dim, 32)
        self.fc2 = LinearBlock(dim, out_dim, activation=False)

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
