#!/bin/bash

function install_opencv3()  {

    # Install requirements
    sudo apt install -y \
	 libglew-dev \
	 libtiff5-dev \
	 zlib1g-dev \
	 libjpeg-dev \
	 libpng12-dev \
	 libjasper-dev \
	 libavcodec-dev \
	 libavformat-dev \
	 libavutil-dev \
	 libpostproc-dev \
	 libswscale-dev \
	 libeigen3-dev \
	 libtbb-dev \
	 libgtk2.0-dev \
	 libv4l-dev \
	 cmake \
	 git \
	 pkg-config \
	 python3-dev \
	 python3-numpy \
	 python3-py\
	 python3-pytest

    # Get the repositories if we don't have them
    if ! [[ -e opencv ]] ; then
	git clone https://github.com/opencv/opencv.git
    fi
    if ! [[ -e opencv_extra ]] ; then
	git clone https://github.com/opencv/opencv_extra.git
    fi

    # Make sure we have the right branch
    cd opencv
    git checkout v3.2.0
    cd -
    cd opencv_extra
    git checkout v3.2.0
    cd -

    # Build
    mkdir opencv/build
    cd opencv/build
    cmake \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX=/usr \
	-DBUILD_PNG=OFF \
	-DBUILD_TIFF=OFF \
	-DBUILD_TBB=OFF \
	-DBUILD_JPEG=OFF \
	-DBUILD_JASPER=OFF \
	-DBUILD_ZLIB=OFF \
	-DBUILD_EXAMPLES=ON \
	-DBUILD_opencv_java=OFF \
	-DBUILD_opencv_python2=OFF \
	-DBUILD_opencv_python3=ON \
	-DPYTHON_INCLUDE_DIR=/usr/include/python3.5m \
	-DPYTHON_INCLUDE_DIR2=/usr/include/aarch64-linux-gnu/python3.5m \
	-DPYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.5m.so \
	-DPYTHON3_NUMPY_INCLUDE_DIRS=/usr/lib/python3/dist-packages/numpy/core/include \
	-DENABLE_PRECOMPILED_HEADERS=OFF \
	-DWITH_OPENCL=OFF \
	-DWITH_OPENMP=OFF \
	-DWITH_FFMPEG=ON \
	-DWITH_GSTREAMER=ON \
	-DWITH_GSTREAMER_0_10=OFF \
	-DWITH_CUDA=ON \
	-DWITH_GTK=ON \
	-DWITH_VTK=OFF \
	-DWITH_TBB=ON \
	-DWITH_1394=OFF \
	-DWITH_OPENEXR=OFF \
	-DWITH_LIBV4L=ON \
	-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0 \
	-DCUDA_ARCH_BIN=6.2 \
	-DCUDA_ARCH_PTX="" \
	-DINSTALL_C_EXAMPLES=OFF \
	-DINSTALL_TESTS=OFF \
	../
    make -j$(nproc)
    sudo make install
    cd -
}

function install_tensorflow() {

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
    sudo pip3 install tensorflow*-cp3*linux_aarch64.whl
}

# Prepare by updating and upgrading all existing packages
sudo apt update
sudo apt upgrade -y

# Install required software
install_opencv3
install_tensorflow
