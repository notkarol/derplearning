import torch.nn as nn
from derp.models.blocks import ConvBlock, LinearBlock, ViewBlock

class PNModel(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(PNModel, self).__init__()
        dim = in_dim.copy()
        self.c1 = ConvBlock(dim, 24, 5, stride=2, padding=0)
        self.c2 = ConvBlock(dim, 36, 5, stride=2, padding=0)
        self.c3 = ConvBlock(dim, 48, 5, stride=2, padding=0)
        self.c4 = ConvBlock(dim, 64, 3, stride=1, padding=0)
        self.c5 = ConvBlock(dim, 64, 3, stride=1, padding=0)
        self.view = ViewBlock(dim)
        self.fc1 = LinearBlock(dim, 100)
        self.fc2 = LinearBlock(dim, 50)
        self.fc3 = LinearBlock(dim, 10)
        self.fc4 = LinearBlock(dim, out_dim, activation=False)

    def forward(self, x):
        out = self.c1(x)
        out = self.c2(out)
        out = self.c3(out)
        out = self.c4(out)
        out = self.c5(out)
        out = self.view(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out
