#!/bin/bash

# Install prereqs
sudo apt install -y \
     zip \
     unzip \
     autoconf \
     automake \
     libtool \
     curl \
     zlib1g-dev \
     maven \
     python3-numpy \
     python3-dev \
     python3-pip \
     python3-wheel \
     swig

if [[ -z $(ls tensorfllow*aarch64.whl) ]] ; then

    # Move cudnn into the right folder
    sudo mkdir /usr/lib/aarch64-linux-gnu/include/
    sudo cp /usr/include/cudnn.h /usr/lib/aarch64-linux-gnu/include/cudnn.h
    
    # Install Java
    sudo add-apt-repository ppa:webupd8team/java
    sudo apt update
    sudo apt install -y oracle-java8-installer

    # Install bazel
    wget --no-certificate-check https://github.com/bazelbuild/bazel/releases/download/0.5.2/bazel-0.5.2-dist.zip
    mkdir bazel
    cd bazel
    unzip ../bazel-0.5.2-dist.zip
    ./compile.sh
    sudo cp output/bazel /usr/local/bin
    cd -
    rm -rf bazel

    # create swap file
    fallocate -l 8G swapfile
    chmod 600 swapfile
    mkswap swapfile
    sudo swapon swapfile
    swapon -s

    # Get tensorflow
    if ! [[ -e tensorflow ]] ; then
	git clone https://github.com/tensorflow/tensorflow.git
    fi
    cd tensorflow
    git checkout v1.3.0-rc1

    # Fix numa stuff for ARM
    sed -i 's/static int TryToReadNumaNode(const string \&pci_bus_id, int device_ordinal) {/static int TryToReadNumaNode(const string \&pci_bus_id, int device_ordinal) {\n  LOG(INFO) << "ARM has no NUMA node, hardcoding to return zero";\n  return 0;/g' tensorflow/stream_executor/cuda/cuda_gpu_executor.cc

    # Replace eigen version
    sed -i 's/f3a22f35b044/d781c1de9834/g' tensorflow/workspace.bzl
    sed -i 's/ca7beac153d4059c02c8fc59816c82d54ea47fe58365e8aded4082ded0b820c4/a34b208da6ec18fa8da963369e166e4a368612c14d956dd2f9d7072904675d9b/g' tensorflow/workspace.bzl
    
    # configure. default settings except to use python3, cuda, and compute capability to 5.3,6.2
    ./configure

    # Build tensorflow
    bazel build --config=opt --config=cuda --local_resources 3072,4.0,1.0 --verbose_failures //tensorflow/tools/pip_package:build_pip_package
    
    # Build pip package
    bazel-bin/tensorflow/tools/pip_package/build_pip_package $PWD/build_wheel

    # Move wheel to local directory and install
    cd ..
    mv tensorflow/build_wheel/tensorflow*-cp3*linux_aarch64.whl .

    # remove swap
    sudo swapoff swapfile
    sudo rm swapfile
fi

# Install 
sudo pip3 install tensorflow*-cp*linux_aarch64.whl
