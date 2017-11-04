
from derp.model.convblock import ConvBlock
from derp.model.linearblock import LinearBlock
from derp.model.viewblock import ViewBlock

class BasicModel(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(BasicModel, self).__init__()
        dim = in_dim.copy()
        self.c1 = ConvBlock(dim, 32, 5, stride=2)
        self.c2 = ConvBlock(dim, 48, 3, pool='max')
        self.c3 = ConvBlock(dim, 64, 3, pool='max')
        self.c4 = ConvBlock(dim, 80, 3, pool='max')
        self.view = ViewBlock(dim)
        self.fc1 = nn.Linear(dim, 64)
        self.fc2 = nn.Linear(dim, out_dim, activation=False)

    def forward(self, x):
        out = self.c1(x)
        out = self.c2(out)
        out = self.c3(out)
        out = self.c4(out)
        out = self.view(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
