#!/usr/bin/bash

version=pycuda-2018.1.1

sudo apt-get install -y \
     build-essential \
     python3-dev \
     python3-setuptools \
     libboost-python-dev \
     libboost-thread-dev

wget https://files.pythonhosted.org/packages/58/33/cced4891eddd1a3ac561ff99081019fddc7838a07cace272c941e3c2f915/pycuda-2018.1.1.tar.gz

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

