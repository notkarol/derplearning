#!/bin/bash

# Make sure we're working off of latest packages
sudo apt update
sudo apt upgrade -y
sudo apt install \
     libusb-1.0-0-dev \
     python3-h5py
sudo apt install python-evdev

# Install Python Packages
pip3 install --user -U pyusb

# Make sure we can communicate to Micro Maestro
sudo cp 99-pololu.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules

# Compile ffmepg from source with GPU support
bash ffmpeg.sh

# Compile OpenCV3 from source
bash opencv3.sh

# Compile PyTorch from source
bash pytorch.sh
