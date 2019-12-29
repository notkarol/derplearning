"""
A collection of neural network code. The first part of the script includes
blocks, which are the building blocks of our models. The second part includes
the actual Pytorch models.
"""
import torch


class ConvBlock(torch.nn.Module):
    """
    A ConvBlock represents a convolution. It's not just a convolution however,
    as some common operations (dropout, activation, batchnorm, 2x2 pooling)
    can be set and run in the order mentioned.
    """

    def __init__(self, dim, n_out, kernel_size=3, stride=1, padding=1, batchnorm=False,
                 dropout=0, activation=True):
        """ A convolution operation """
        super(ConvBlock, self).__init__()
        n_in = int(dim[0])
        self.conv2d = torch.nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                                      stride=stride, padding=padding)
        self.batchnorm = torch.nn.BatchNorm2d(n_out) if batchnorm else None
        self.activation = torch.nn.ReLU(inplace=True) if activation else None
        self.dropout = torch.nn.Dropout2d(dropout) if dropout else None
        dim[0] = n_out
        dim[1:] = 1 + (dim[1:] + padding * 2 - kernel_size) // stride
        self.n_params = n_out * (n_in * kernel_size * kernel_size + (3 if batchnorm else 1))
        print("Conv2d in %4i out %4i h %4i w %4i k %i s %i params %9i"
              % (n_in, *dim, kernel_size, stride, self.n_params))

    def forward(self, batch):
        """ Forward the 4D batch """
        out = self.conv2d(batch)
        if self.activation:
            out = self.activation(out)
        if self.batchnorm:
            out = self.batchnorm(out)
        if self.dropout:
            out = self.dropout(out)
        return out


class LinearBlock(torch.nn.Module):
    """
    A LinearBlock represents a fully connected layer. It's not just this, as
    some common operations (dropout, activation, batchnorm) can be set and run
    in the order mentioned.
    """

    def __init__(self, dim, n_out, batchnorm=False, dropout=0.0, activation=True):
        """ A fully connected operation """
        super(LinearBlock, self).__init__()
        n_in = int(dim[0])
        self.linear = torch.nn.Linear(n_in, n_out)
        dim[0] = n_out if type(n_out) in (int, float) else n_out[0]
        self.batchnorm = torch.nn.BatchNorm1d(dim[0]) if batchnorm else None
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else None
        self.activation = torch.nn.ReLU(inplace=True) if activation else None
        self.n_params = n_out * (n_in + (3 if batchnorm else 1))
        print("Linear in %4i out %4i                       params %9i"
              % (n_in, n_out, self.n_params))

    def forward(self, batch):
        """ Forward the 2D batch """
        out = self.linear(batch)
        if self.activation:
            out = self.activation(out)
        if self.batchnorm:
            out = self.batchnorm(out)
        if self.dropout:
            out = self.dropout(out)
        return out
                

class PoolBlock(torch.nn.Module):
    """
    A PoolBlock is a pooling operation that happens on a matrix, often between
    convolutional layers, on each channel individually. By default only two are
    supported: max and avg.
    """

    def __init__(self, dim, pool="max", size=None, stride=None):
        """ A pooling operation """
        super(PoolBlock, self).__init__()

        stride = size if stride is None else stride
        if size:
            dim[1:] //= stride
        else:
            size = [int(x) for x in dim[1:]]
            dim[1:] = 1
        if pool == "max":
            self.pool = torch.nn.MaxPool2d(size, stride=stride, padding=0)
        elif pool == "avg":
            self.pool = torch.nn.AvgPool2d(size, stride=stride, padding=0)
        self.n_params = 0

    def forward(self, batch):
        """ Forward the 4D batch """
        out = self.pool(batch)
        return out


class ViewBlock(torch.nn.Module):
    """
    A ViewBlock restructures the shape of our activation maps so they're
    represented as 1D instead of 3D.
    """
    def __init__(self, dim, shape=-1):
        """ A reshape operation """
        super(ViewBlock, self).__init__()
        self.shape = shape
        if self.shape == -1:
            dim[0] = dim[0] * dim[1] * dim[2]
            dim[-2] = 0
            dim[-1] = 0
        else:
            dim[:] = shape

        self.n_params = 0
        print("View             d %4i h %4i w %4i" % (*dim,))

    def forward(self, batch):
        """ Forward the 4D batch into a 2D batch """
        return batch.view(batch.size(0), self.shape)


class Tiny(torch.nn.Module):
    """ A small and quick model """

    def __init__(self, in_dim, n_status, n_out):
        """
        Args:
            in_dim (list): The input size of each example
            n_status (int): Number of status inputs to add
            n_out (int): Number of values to predict
        """
        super(Tiny, self).__init__()
        self.n_status = n_status
        dim = in_dim.copy()
        self.feat = torch.nn.Sequential(
            ConvBlock(dim, 16),
            PoolBlock(dim, 'max', 2),
            ConvBlock(dim, 32),
            PoolBlock(dim, 'max', 2),
            ConvBlock(dim, 48),
            PoolBlock(dim, 'max', 2),
            ConvBlock(dim, 64),
            PoolBlock(dim, 'max', 2),
        )
        self.view = ViewBlock(dim)
        dim[0] += n_status
        self.head = torch.nn.Sequential(LinearBlock(dim, n_out, activation=False))
        self.n_params = sum([x.n_params for x in self.feat]) + sum([x.n_params for x in self.head])
        print("Tiny                                          params %9i" % self.n_params)

    def forward(self, batch, status):
        """
        Args:
            batch (4D tensor): A batch of camera input.
            status (1D tensor): Status inputs indicating things like speed.
        """
        out = self.feat(batch)
        out = self.view(out)
        if self.n_status:
            out = torch.cat((out, status), 1)
        out = self.head(out)
        return out


class StarTree(torch.nn.Module):
    """
    A medium-sized model that uses layers with few activation maps to
    efficiently increase the number of layers, and therefore nonlinearities.
    """

    def __init__(self, in_dim, n_status, n_out):
        """
        Args:
            in_dim (list): The input size of each example
            n_status (int): Number of status inputs to add
            n_out (int): Number of values to predict
        """
        super(StarTree, self).__init__()
        self.n_status = n_status
        dim = in_dim.copy()
        self.feat = torch.nn.Sequential(
            ConvBlock(dim, 64, dropout=0.25),
            ConvBlock(dim, 16),
            ConvBlock(dim, 32),
            PoolBlock(dim, 'max', 2),
            ConvBlock(dim, 24),
            ConvBlock(dim, 48),
            PoolBlock(dim, 'max', 2),
            ConvBlock(dim, 32),
            ConvBlock(dim, 64),
            PoolBlock(dim, 'max', 2),
            ConvBlock(dim, 40),
            ConvBlock(dim, 80, dropout=0.25),
            PoolBlock(dim, 'max', 2),
        )
        self.view = ViewBlock(dim)
        dim[0] += n_status
        self.head = torch.nn.Sequential(
            LinearBlock(dim, 50),
            LinearBlock(dim, n_out, activation=False),
        )
        self.n_params = sum([x.n_params for x in self.feat]) + sum([x.n_params for x in self.head])
        print("StarTree                                  params %9i" % self.n_params)

    def forward(self, batch, status):
        """
        Args:
            batch (4D tensor): A batch of camera input.
            status (1D tensor): Status inputs indicating things like speed.
        """
        out = self.feat(batch)
        out = self.view(out)
        if self.n_status:
            out = torch.cat((out, status), 1)
        out = self.head(out)
        return out
