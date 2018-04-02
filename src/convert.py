import argparse
from matplotlib.pyplot import imshow
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import tensorrt as trt
import derp.util


def add_bb(network, weights, name, prev, n_out, kernel_size,
           stride, padding, batchnorm, pool):
    w = weights['%s.conv2d.weight' % name].cpu().numpy().reshape(-1)
    b = weights['%s.conv2d.bias' % name].cpu().numpy().reshape(-1)
    c = network.add_convolution(prev, n_out, kernel_size, w, b)
    assert(c)
    c.set_stride(stride)
    c.set_padding(padding)

    # activation
    a = network.add_activation(c.get_output(0), trt.infer.ActivationType.RELU)
    assert(a)
    
    # batchnorm scale # TODO
    s = a
    
    # we're done if we don't pool
    if pool:
        p = network.add_pooling(a.get_output(0), trt.infer.PoolingType.MAX, (2,2))
        assert(p)
    else:
        p = a    
    return p


def add_fc(network, weights, name, prev, n_out, activation):
    w = weights['%s.linear.weight' % name].cpu().numpy().reshape(-1)
    b = weights['%s.linear.bias' % name].cpu().numpy().reshape(-1)
    f = network.add_fully_connected(prev, n_out, w, b)
    assert(f)

    if activation:
        a = network.add_activation(f.get_output(0), trt.infer.ActivationType.RELU)
        assert(a)
    else:
        a = f

    return a


def add_cc(network, left, right):
    # TODO, tensorrt documentation for this is incomplete
    return network.add_concatenation((left, right), 2)


def build_network(weights, builder):
    network = builder.create_network()

    # Prepare data input size
    data = network.add_input("data", trt.infer.DataType.FLOAT, (3, 64, 128))

    c1a = add_bb(network, weights, 'c1a', data, 12, (5,5), (2,2), (2,2), False, False)
    c2a = add_bb(network, weights, 'c2a', c1a.get_output(0),  16, (3,3), (1,1), (1,1), False, True)
    c3a = add_bb(network, weights, 'c3a', c2a.get_output(0),  20, (3,3), (1,1), (1,1), False, True)
    c4a = add_bb(network, weights, 'c4a', c3a.get_output(0),  24, (3,3), (1,1), (1,1), False, True)
    c5a = add_bb(network, weights, 'c5a', c4a.get_output(0),  28, (3,3), (1,1), (1,1), False, True)
    c6a = add_bb(network, weights, 'c6a', c5a.get_output(0),  32, (2,2), (1,1), (0,0), False, False)

    fc1 = add_fc(network, weights, 'fc1', c6a.get_output(0), 32, True)
    
    fc2 = add_fc(network, weights, 'fc2', fc1.get_output(0), 32, True)
    fc3 = add_fc(network, weights, 'fc3', fc2.get_output(0), 2, False)

    fc3.get_output(0).set_name("predictions")
    network.mark_output(fc3.get_output(0))

    return network

def build_engine(weights):
    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
    builder = trt.infer.create_infer_builder(G_LOGGER)

    network = build_network(weights, builder)

    builder.set_max_batch_size(1)
    builder.set_max_workspace_size(1 << 31)
    
    engine = builder.build_cuda_engine(network)
    network.destroy()
    builder.destroy()

    return engine

def main(args):

    model = torch.load(args.model_path)

    # Prepare input
    image = np.zeros((3, 64, 128), dtype=np.float32)
    flat_image = image.ravel()
    tensorrt_out = np.empty(2, dtype=np.float32)
    
    # Prepare torch version
    model.eval()
    image_shape = tuple([1] + list(image.shape))
    torch_image = np.reshape(image, image_shape)
    batch = Variable(torch.from_numpy(torch_image).cuda() / 255.0)
    torch_out = model.forward(batch, None).cpu().data.numpy()[0]    
    print("Torch Predictions:", torch_out)
    weights = model.state_dict()
    engine = build_engine(weights)
    context = engine.create_execution_context()
    d_input = cuda.mem_alloc(1 * flat_image.size * flat_image.dtype.itemsize)
    d_output = cuda.mem_alloc(1 * tensorrt_out.size * tensorrt_out.dtype.itemsize)
    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()

    #transfer input data to device
    cuda.memcpy_htod_async(d_input, flat_image, stream)
    
    #execute model
    context.enqueue(1, bindings, stream.handle, None)
    #transfer predictions back
    cuda.memcpy_dtoh_async(tensorrt_out, d_output, stream)
    #syncronize threads
    stream.synchronize()

    print("TensorRT Predictions: ", tensorrt_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()
    main(args)
