#!/bin/bash

# Install latest python pip
sudo apt-get install python3-pip
pip3 install -U pip

# Get pytorch
if ! [[ -e pytorch ]] ; then
    git clone https://github.com/pytorch/pytorch.git
    cd pytorch
    git submodule --update --init
else
    cd pytorch
    git pull
    git submodule --update    
fi

sudo pip3 install setuptools
sudo pip3 install -r requirements.txt

# Build dependencies and install. This might need to be restarted
python3 setup.py build_deps
sudo python3 setup.py install
