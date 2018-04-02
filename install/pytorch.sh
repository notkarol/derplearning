#!/bin/bash

PYTORCH_VERSION=v0.3.0

# Install latest python pip and setuptools and openblas
sudo apt-get install libopenblas-dev
pip3 install --user --upgrade pip setuptools

# Pytorch
if ! [[ -e pytorch ]] ; then
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch
else
    cd pytorch
    git pull
    git submodule --update    
fi

git checkout ${PYTORCH_VERSION}
pip3 install --user -r requirements.txt
python3 setup.py install --user


# Torch Vision
if ! [[ -e vision ]] ; then
    git clone --recursive  https://github.com/pytorch/vision
    cd vision
else
    cd vision
    git pull
fi
pip3 install --user -r requirements.txt
python3 setup.py install --user
cd -

# Cleanup after ourselves
rm -rf pytorch vision
