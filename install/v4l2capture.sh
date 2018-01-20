#!/bin/bash

sudo apt install -y libv4l2rds0 libv4l-dev libv4l-0 v4l-utils

if ! [[ -e "python-v4l2capture" ]] ; then
    git clone https://github.com/gebart/python-v4l2capture.git
fi

cd python-v4l2capture
python3 setup.py install --user
cd -

# Cleanup after ourselves
rm -rf python-v4l2capture

