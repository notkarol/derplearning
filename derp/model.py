import torch
import torch.nn as nn
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, dim, n_out, kernel_size=3, stride=1,
                 padding=None, batchnorm=True,
                 activation=True, pool=None,
                 dropout=0.0, verbose=False):
        super(ConvBlock, self).__init__()

        if padding is None:
            padding = kernel_size // 2

        n_in = int(dim[0])
        self.conv2d = nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                                stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(n_out) if batchnorm else None
        self.activation = nn.ReLU(inplace=True) if activation else None
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else None

        # Output dimensions just from convolution
        dim[0] = n_out
        dim[1:] = 1 + np.floor((dim[1:] + padding * 2 - kernel_size) / stride)

        # Add a pooling block, which might further change dim
        self.pool = pool if pool is None else PoolBlock(dim, pool, 2, verbose=verbose)
        
        # Number of params
        self.n_params = n_out * (n_in * kernel_size * kernel_size + (3 if batchnorm else 1))

        if verbose:
            print("Conv2d in %3i out %3i h %3i w %3i k %i s %i params %9i" %
                  (n_in, *dim, kernel_size, stride, self.n_params))


    def forward(self, x):
        out = self.conv2d(x)
        if self.dropout:
            out = self.dropout(out)
        if self.activation is not None:
            out = self.activation(out)
        if self.batchnorm is not None:
            out = self.batchnorm(out)
        if self.pool is not None:
            out = self.pool(out)
        return out


class ResnetBlock(nn.Module):
    def __init__(self, dim, n_out, kernel_size=3, stride=1, pool=None, verbose=False):
        super(ResnetBlock, self).__init__()

        n_in = int(dim[0])
        residual_dim = dim.copy() # keep a copy since we're going down parallel channels
        self.c1 = ConvBlock(dim, n_out, kernel_size, stride, pool=pool, verbose=verbose)
        self.c2 = ConvBlock(dim, n_out, kernel_size, activation=False, verbose=verbose)
        if n_in != n_out:
            self.c3 = ConvBlock(residual_dim, n_out, 1, stride, pool=pool, verbose=verbose)
            self.pool = None
        else:
            self.c3 = None
            self.pool = pool if pool is None else PoolBlock(residual_dim, pool, 2)
        self.activation = nn.ReLU(inplace=True) 

        self.n_params = self.c1.n_params = self.c2.n_params
        if verbose:
            pass # use convolutions


    def forward(self, x):

        # Residual
        residual = x
        if self.pool is not None:
            residual = self.pool(residual)
        if self.c3 is not None:
            residual = self.c3(residual)

        # Parallel
        out = self.c1(x)
        out = self.c2(out)
        out += residual
        out = self.activation(out)
        return out

class LinearBlock(nn.Module):
    def __init__(self, dim, n_out, dropout=0.0, bn=False, activation=True, verbose=False):
        super(LinearBlock, self).__init__()
        n_in = int(dim[0])
        self.linear = nn.Linear(n_in, n_out)
        dim[0] = n_out if type(n_out) in (int, float) else n_out[0]
        self.batchnorm = nn.BatchNorm(dim[0]) if bn else None
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.activation = nn.ReLU(inplace=True) if activation else None

        # Number of params, add 3 becuase of conv bias and mean/std for batchnorm
        self.n_params = n_out * (n_in + (3 if bn else 2))

        if verbose:
            print("Linear in %3i out %3i                     params %9i" %
                  (n_in, n_out, self.n_params))

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
    def __init__(self, dim, pool='max', size=None, stride=None, verbose=False):
        super(PoolBlock, self).__init__()

        stride = size if stride is None else stride
        if size is not None:
            dim[1:] = np.floor(dim[1:] / stride)
        else:
            size = [int(x) for x in dim[1:]]
            dim[1:] = 1
        if pool == 'max':
            self.pool = nn.MaxPool2d(size, size)
        elif pool == 'avg':
            self.pool = nn.AvgPool2d(size, size)
            
        self.n_params = 0
        if verbose:
            pass
            
    def forward(self, x):
        out = self.pool(x)
        return out


class ViewBlock(nn.Module):
    def __init__(self, dim, shape=-1, verbose=False):
        super(ViewBlock, self).__init__()
        self.shape = shape
        if self.shape == -1:
            dim[0] = dim[0] * dim[1] * dim[2]
            dim[-2] = 0 
            dim[-1] = 0
        else:
            dim[:] = shape

        self.n_params = 0
        if verbose:
            print("View            d %3i h %3i w %3i" % (*dim,))
            
    def forward(self, x):
        out = x.view(x.size(0), self.shape)
        return out

class Tiny(nn.Module):
    def __init__(self, in_dim, n_status, n_out, verbose=True):
        super(Tiny, self).__init__()
        self.n_status = n_status
        dim = in_dim.copy()
        self.c1 = ConvBlock(dim, 32, 5, stride=2, batchnorm=False, verbose=verbose)
        while (dim[-1] > 
        self.c2 = ConvBlock(dim, 40, 3, pool='max', batchnorm=False, verbose=verbose)
        self.c3 = ConvBlock(dim, 48, 3, pool='max', batchnorm=False, verbose=verbose)
        self.c4 = ConvBlock(dim, 56, 3, pool='max', batchnorm=False, verbose=verbose)
        self.c5 = ConvBlock(dim, 64, 2, pool='max', batchnorm=False, verbose=verbose)
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


class StarTree(nn.Module):
    def __init__(self, in_dim, n_status, n_out, verbose=True):
        super(StarTree, self).__init__()
        self.n_status = n_status
        dim = in_dim.copy()
        self.c1 = ConvBlock(dim, 64, 5, stride=2, verbose=verbose)
        self.c2 = ConvBlock(dim, 16, 3, verbose=verbose)
        self.c3 = ConvBlock(dim, 32, 3, pool='max', verbose=verbose)
        self.c4 = ConvBlock(dim, 24, 3, verbose=verbose)
        self.c5 = ConvBlock(dim, 48, 3, pool='max', verbose=verbose)
        self.c6 = ConvBlock(dim, 32, 3, verbose=verbose)
        self.c7 = ConvBlock(dim, 64, 3, pool='max', verbose=verbose)
        self.c8 = ConvBlock(dim, 40, 3, verbose=verbose)
        self.c9 = ConvBlock(dim, 80, 2, pool='max', verbose=verbose, dropout=0.25)
        self.view = ViewBlock(dim, verbose=verbose)
        dim[0] += n_status
        self.fc1 = LinearBlock(dim, 50, verbose=verbose)
        self.fc2 = LinearBlock(dim, n_out, activation=False, verbose=verbose)

        self.n_params = sum([self._modules[x].n_params for x in self._modules])
        if verbose:
            print("StarTree                                  params %9i" % self.n_params)
        
    def forward(self, x, status):
        out = self.c1(x)
        out = self.c2(out)
        out = self.c3(out)
        out = self.c4(out)
        out = self.c5(out)
        out = self.c6(out)
        out = self.c7(out)
        out = self.c8(out)
        out = self.c9(out)
        out = self.view(out)
        if self.n_status:
            out = torch.cat((out, status), 1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
