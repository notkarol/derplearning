#!/bin/bash

# Make sure we're working off of latest packages
sudo apt update
sudo apt upgrade -y
sudo apt install \
     libusb-1.0-0-dev \
     python3-h5py

# Install Python Packages
pip3 install --user -U pyusb


# Make sure we can communicate to Micro Maestro
sudo cp install/99-pololu.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules

# Compile ffmepg from source with GPU support
bash install/ffmpeg.sh

# Compile OpenCV3 from source
bash install/opencv3.sh

# Compile PyTorch from source
bash install/pytorch.sh
