#!/bin/bash
PYTORCH_VERSION=v0.2.0

# Install latest python pip and setuptools
sudo apt-get install python3-pip
pip3 install --user --upgrade pip setuptools

# Get pytorch
if ! [[ -e pytorch ]] ; then
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch
else
    cd pytorch
    git pull
    git submodule --update    
fi

git checkout ${PYTORCH_VERSION}

# Install specified requirements
sudo pip3 install -r requirements.txt

# Build dependencies and install. This might need to be restarted
sudo python3 setup.py install
