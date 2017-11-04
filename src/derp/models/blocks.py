import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, dim, n_out, kernel_size=3, stride=1,
                 padding=None, pool=None, dropout=0.0):
        super(ConvBlock, self).__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv2d = nn.Conv2d(dim[0], n_out, kernel_size=kernel_size,
                                stride=stride, padding=padding)
        dim[1:] -= 2 * (kernel_size // 2 - padding)
        dim[1:] //= stride
        self.batchnorm = nn.BatchNorm2d(n_out)
        self.activation = nn.ELU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else None

        self.pool = None
        if pool == 'max':
            self.pool = nn.MaxPool2d(2, stride=2, padding=0)
            dim[1:] //= 2
        elif pool == 'avg':
            self.pool = nn.AvgPool2d(2, stride=2, padding=0)
            dim[1:] //= 2

            
    def forward(self, x):
        out = self.conv2d(x)
        out = self.batchnorm(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.activation(out)
        if self.pool is not None:
            out = self.pool(out)
        return out


class LinearBlock(nn.Module):
    def __init__(self, dim, n_out, dropout=0.0, bn=False, activation=True):
        super(LinearBlock, self).__init__()

        self.linear = nn.Linear(dim[0], n_out)
        dim[0] = n_out if type(n_out) in (int, float) else n_out[0]
        self.batchnorm = nn.BatchNorm(dim[0]) if bn else None
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.activation = nn.ELU(inplace=True) if activation else None
            
    def forward(self, x):
        out = self.linear(x)
        if self.batchnorm is not None:
            out = self.batchnorm(out)
        if self.dropout is not None:
            out = self.dropout(out)
        if self.activation is not None:
            out = self.activation(out)
        return out


class PoolBlock(nn.Module):
    def __init__(self, dim, size=None, stride=None)
        super(PoolBlock, self).__init__()

        if size = None:
            size = dim[-2:]
            dim[-2:] = 1
        elif stride:
            
            
    def forward(self, x):
        out = x.view(x.size(0), self.shape)
        return out

    
class ResnetBlock(nn.Module):
    def __init__(self, dim, n_out, kernel_size=3, stride=1):
        super(ResnetBlock, self).__init__()

        self.c1 = nn.Conv2d(dim[0], n_out, kernel_size=kernel_size, stride=stride,
                            padding=kernel_size // 2, bias=False)
        dim[1:] //= stride
        self.c2 = nn.Conv2d(n_out, n_out, kernel_size=kernel_size, stride=1,
                            padding=kernel_size // 2, bias=False)
        self.pool = None if stride == 1 else nn.AvgPool2D(stride, stride)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.bn2 = nn.BatchNorm2d(n_out)
        self.elu = nn.ELU(inplace=True)
        
    def forward(self, x):
        residual = x
        if self.pool is not None:
            residual = self.pool(residual)
        out = self.c1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.c2(out)
        out = self.bn2(out)
        out += residual
        out = self.elu(out)
        return out


class ViewBlock(nn.Module):
    def __init__(self, dim, shape=-1):
        super(ViewBlock, self).__init__()
        self.shape = shape
        if self.shape == -1:
            dim[-2] = dim[-1] * dim[-2]
            dim[-1] = 0
        else:
            dim[1:] = shape
            
    def forward(self, x):
        out = x.view(x.size(0), self.shape)
        return out
