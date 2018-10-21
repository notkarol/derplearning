#!/bin/bash

PYTORCH_VERSION=0.4.1
if [[ -z $(pip3 freeze | grep "torch==${PYTORCH_VERSION}") ]] ; then
   exit 0
fi   

# Pull in the code if it's not already done so
if ! [[ -e pytorch ]] ; then
    git clone --recursive https://github.com/pytorch/pytorch
fi

# Prepare the right version of the code
cd pytorch
git pull
git checkout v${PYTORCH_VERSION}
git submodule update --init

# Install requirements and then the package
sudo apt-get install libopenblas-dev
pip3 install --user -r requirements.txt
python3 setup.py install --user
cd ..

# Pull in the torchvision code for dataloaders and image manipulation.
if ! [[ -e vision ]] ; then
    git clone --recursive  https://github.com/pytorch/vision
fi

# Install requirements and then the package
cd vision
git pull
python3 setup.py install --user
cd ..

# Cleanup after ourselves
# rm -rf pytorch vision
