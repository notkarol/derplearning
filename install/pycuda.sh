#!/usr/bin/bash

version=pycuda-2017.1.1

sudo apt-get install -y \
     build-essential \
     python3-dev \
     python3-setuptools \
     libboost-python-dev \
     libboost-thread-dev

wget https://pypi.python.org/packages/b3/30/9e1c0a4c10e90b4c59ca7aa3c518e96f37aabcac73ffe6b5d9658f6ef843/${version}.tar.gz

tar xzvf ${version}.tar.gz

cd ${version}

./configure.py \
    --cuda-root=/usr/local/cuda \
    --cudadrv-lib-dir=/usr/lib \
    --boost-inc-dir=/usr/include \
    --boost-lib-dir=/usr/lib \
    --boost-python-libname=boost_python-py35 \
    --boost-thread-libname=boost_thread

make -j4

python3 setup.py install --user

cd ..

rm -rf ${version}
