#!/bin/bash

OPENCV_VERSION=3.4.3
ARCH=$(uname -i)
PYTHON_VERSION=python3.5
CUDA_VERSION=9.0

# Make sure we're installed
python3 -c "import cv2"
if [[ "$?" == "0" ]]
   exit
fi

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
     python-numpy \
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
git checkout ${OPENCV_VERSION}
cd -

cd opencv_extra
git checkout ${OPENCV_VERSION}
cd -

# Build
mkdir opencv/build
cd opencv/build

cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr \
    -D BUILD_PNG=OFF \
    -D BUILD_TIFF=OFF \
    -D BUILD_TBB=OFF \
    -D BUILD_JPEG=OFF \
    -D BUILD_JASPER=OFF \
    -D BUILD_ZLIB=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_opencv_java=OFF \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=ON \
    -D PYTHON_INCLUDE_DIR=/usr/include/${PYTHON_VERSION}m \
    -D PYTHON_INCLUDE_DIR2=/usr/include/${ARCH}-linux-gnu/${PYTHON_VERSION}m \
    -D PYTHON_LIBRARY=/usr/lib/${ARCH}-linux-gnu/lib${PYTHON_VERSION}m.so \
    -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/lib/python3/dist-packages/numpy/core/include \
    -D ENABLE_PRECOMPILED_HEADERS=OFF \
    -D ENABLE_CXX11=ON \
    -D WITH_OPENCL=OFF \
    -D WITH_OPENMP=OFF \
    -D WITH_FFMPEG=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_GSTREAMER_0_10=OFF \
    -D WITH_CUDA=ON \
    -D WITH_GTK=ON \
    -D WITH_VTK=OFF \
    -D WITH_TBB=ON \
    -D WITH_1394=OFF \
    -D WITH_OPENEXR=OFF \
    -D WITH_LIBV4L=ON \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-${CUDA_VERSION} \
    -D CUDA_ARCH_BIN=5.3,6.2 \
    -D CUDA_ARCH_PTX="" \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_TESTS=OFF \
    ../

make -j$(nproc)
sudo make install
cd -

# Cleanup after ourselves
#rm -rf opencv opencv_extra
