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

    def __init__(
        self,
        dim,
        n_out,
        kernel_size=3,
        stride=1,
        padding=None,
        batchnorm=True,
        activation=True,
        pool=None,
        dropout=0.0,
        verbose=False,
    ):
        """ A convolution operation """
        super(ConvBlock, self).__init__()

        if padding is None:
            padding = kernel_size // 2

        n_in = int(dim[0])
        self.conv2d = torch.nn.Conv2d(
            n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.batchnorm = torch.nn.BatchNorm2d(n_out) if batchnorm else None
        self.activation = torch.nn.ReLU(inplace=True) if activation else None
        self.dropout = torch.nn.Dropout2d(dropout) if dropout > 0.0 else None

        # Output dimensions just from convolution
        dim[0] = n_out
        dim[1:] = 1 + (dim[1:] + padding * 2 - kernel_size) // stride

        # Add a pooling block, which might further change dim
        self.pool = pool if pool is None else PoolBlock(dim, pool, 2, verbose=verbose)

        # Number of params
        self.n_params = n_out * (n_in * kernel_size * kernel_size + (2 if batchnorm else 1))

        if verbose:
            print(
                "Conv2d in %3i out %3i h %3i w %3i k %i s %i params %9i"
                % (n_in, *dim, kernel_size, stride, self.n_params)
            )

    def forward(self, batch):
        """ Forward the 4D batch """
        out = self.conv2d(batch)
        if self.dropout:
            out = self.dropout(out)
        if self.activation:
            out = self.activation(out)
        if self.batchnorm:
            out = self.batchnorm(out)
        if self.pool:
            out = self.pool(out)
        return out


class ResnetBlock(torch.nn.Module):
    """
    A Resnet block is a special kind of chained convolution that learns to
    progressively update a manifold in each channel, instead of replacing it.
    """

    def __init__(self, dim, n_out, kernel_size=3, stride=1, pool=None, verbose=False):
        """ A residual convolution operation """
        super(ResnetBlock, self).__init__()

        n_in = int(dim[0])
        residual_dim = dim.copy()  # keep a copy since we're going down parallel channels
        self.conv1 = ConvBlock(dim, n_out, kernel_size, stride, pool=pool, verbose=verbose)
        self.conv2 = ConvBlock(dim, n_out, kernel_size, activation=False, verbose=verbose)
        if n_in != n_out:
            self.conv3 = ConvBlock(residual_dim, n_out, 1, stride, pool=pool, verbose=verbose)
            self.pool = None
        else:
            self.conv3 = None
            self.pool = pool if pool is None else PoolBlock(residual_dim, pool, 2)
        self.activation = torch.nn.ReLU(inplace=True)

        self.n_params = (
            self.conv1.n_params + self.conv2.n_params + (self.conv3.params if self.conv3 else 0)
        )

    def forward(self, batch):
        """ Forward the 4D batch """
        residual = batch
        if self.pool:
            residual = self.pool(residual)
        if self.conv3:
            residual = self.conv3(residual)

        out = self.conv1(batch)
        out = self.conv2(out)
        out += residual
        out = self.activation(out)
        return out


class LinearBlock(torch.nn.Module):
    """
    A LinearBlock represents a fully connected layer. It's not just this, as
    some common operations (dropout, activation, batchnorm) can be set and run
    in the order mentioned.
    """

    def __init__(self, dim, n_out, dropout=0.0, batchnorm=False, activation=True, verbose=False):
        """ A fully connected operation """
        super(LinearBlock, self).__init__()
        n_in = int(dim[0])
        self.linear = torch.nn.Linear(n_in, n_out)
        dim[0] = n_out if type(n_out) in (int, float) else n_out[0]
        self.batchnorm = torch.nn.BatchNorm1d(dim[0]) if batchnorm else None
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else None
        self.activation = torch.nn.ReLU(inplace=True) if activation else None

        # Number of params, add 3 becuase of conv bias and mean/std for batchnorm
        self.n_params = n_out * (n_in + (3 if batchnorm else 2))

        if verbose:
            print(
                "Linear in %3i out %3i                     params %9i"
                % (n_in, n_out, self.n_params)
            )

    def forward(self, batch):
        """ Forward the 2D batch """
        out = self.linear(batch)
        if self.batchnorm:
            out = self.batchnorm(out)
        if self.dropout:
            out = self.dropout(out)
        if self.activation:
            out = self.activation(out)
        return out


class PoolBlock(torch.nn.Module):
    """
    A PoolBlock is a pooling operation that happens on a matrix, often between
    convolutional layers, on each channel individually. By default only two are
    supported: max and avg.
    """

    def __init__(self, dim, pool="max", size=None, stride=None, verbose=False):
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

    def __init__(self, dim, shape=-1, verbose=False):
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
        if verbose:
            print("View            d %3i h %3i w %3i" % (*dim,))

    def forward(self, batch):
        """ Forward the 4D batch into a 2D batch """
        out = batch.view(batch.size(0), self.shape)
        return out


class Tiny(torch.nn.Module):
    """ A small and quick model """

    def __init__(self, in_dim, n_status, n_out, verbose=True):
        """
        Args:
            in_dim (list): The input size of each example
            n_status (int): Number of status inputs to add
            n_out (int): Number of values to predict
            verbose (bool): Whether to print the network architecture
        """
        super(Tiny, self).__init__()
        self.n_status = n_status
        dim = in_dim.copy()
        self.feat = torch.nn.Sequential(
            ConvBlock(dim, 32, 5, stride=2, batchnorm=False, verbose=verbose),
            ConvBlock(dim, 32, 3, pool="max", batchnorm=False, verbose=verbose),
            ConvBlock(dim, 32, 3, pool="max", batchnorm=False, verbose=verbose),
            ConvBlock(dim, 32, 2, pool="max", batchnorm=False, verbose=verbose),
        )
        self.view = ViewBlock(dim, verbose=verbose)
        dim[0] += n_status
        self.head = torch.nn.Sequential(LinearBlock(dim, n_out, activation=False, verbose=verbose))
        self.n_params = sum([x.n_params for x in self.feat]) + sum([x.n_params for x in self.head])
        if verbose:
            print("Tiny                                      params %9i" % self.n_params)

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

    def __init__(self, in_dim, n_status, n_out, verbose=True):
        """
        Args:
            in_dim (list): The input size of each example
            n_status (int): Number of status inputs to add
            n_out (int): Number of values to predict
            verbose (bool): Whether to print the network architecture
        """
        super(StarTree, self).__init__()
        self.n_status = n_status
        dim = in_dim.copy()
        self.feat = torch.nn.Sequential(
            ConvBlock(dim, 64, 5, stride=2, verbose=verbose),
            ConvBlock(dim, 16, 3, verbose=verbose),
            ConvBlock(dim, 32, 3, pool="max", verbose=verbose),
            ConvBlock(dim, 24, 3, verbose=verbose),
            ConvBlock(dim, 48, 3, pool="max", verbose=verbose),
            ConvBlock(dim, 32, 3, verbose=verbose),
            ConvBlock(dim, 64, 3, pool="max", verbose=verbose),
            ConvBlock(dim, 40, 3, verbose=verbose),
            ConvBlock(dim, 80, 2, pool="max", verbose=verbose, dropout=0.25),
        )
        self.view = ViewBlock(dim, verbose=verbose)
        dim[0] += n_status
        self.head = torch.nn.Sequential(
            LinearBlock(dim, 50, verbose=verbose),
            LinearBlock(dim, n_out, activation=False, verbose=verbose),
        )
        self.n_params = sum([x.n_params for x in self.feat]) + sum([x.n_params for x in self.head])
        if verbose:
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


class Resnet20(torch.nn.Module):
    """
    A large model that is based on residual connections to process information.
    """

    def __init__(self, in_dim, n_status, n_out, verbose=True):
        """
        Args:
            in_dim (list): The input size of each example
            n_status (int): Number of status inputs to add
            n_out (int): Number of values to predict
            verbose (bool): Whether to print the network architecture
        """
        super(Resnet20, self).__init__()
        self.n_status = n_status
        dim = in_dim.copy()
        self.feat = torch.nn.Sequential(
            ConvBlock(dim, 64, 5, stride=2, verbose=verbose),
            ResnetBlock(dim, 64, 3, verbose=verbose),
            ResnetBlock(dim, 64, 3, verbose=verbose),
            ResnetBlock(dim, 64, 3, verbose=verbose),
            ResnetBlock(dim, 96, 3, stride=2, verbose=verbose),
            ResnetBlock(dim, 96, 3, verbose=verbose),
            ResnetBlock(dim, 96, 3, verbose=verbose),
            ResnetBlock(dim, 96, 3, verbose=verbose),
            ResnetBlock(dim, 128, 3, stride=2, verbose=verbose),
            ConvBlock(dim, 128, 1, pool="max", verbose=verbose),
        )
        self.view = ViewBlock(dim, verbose=verbose)
        dim[0] += n_status
        self.head = torch.nn.Sequential(
            LinearBlock(dim, 50, verbose=verbose),
            LinearBlock(dim, n_out, activation=False, verbose=verbose),
        )
        self.n_params = sum([x.n_params for x in self.feat]) + sum([x.n_params for x in self.head])
        if verbose:
            print("Resnet20                                  params %9i" % self.n_params)

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
