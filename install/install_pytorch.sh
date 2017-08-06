#!/bin/bash

# Install latest python pip
sudo apt-get install python3-pip
pip3 install -U pip

# Get pytorch
if ! [[ -e pytorch ]] ; then
    git clone https://github.com/pytorch/pytorch.git
fi
cd pytorch

# Install requirements
sudo pip3 install -U setuptools
sudo pip3 install -r requirements

# Build dependencies and install. This might need to be restarted
python3 setup.py build_deps
sudo python3 setup.py install
