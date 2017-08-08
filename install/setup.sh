#!/bin/bash

# Make sure we're working off of latest packages
sudo apt update
sudo apt upgrade -y

# Make sure we can communicate to Micro Maestro
pip3 install --user -U pyusb
sudo apt install libusb-1.0-0-dev
sudo cp 99-pololu.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules

# Install necessary software
./install_opencv3.sh
./install_tensorflow.sh
#./install_pytorch.sh
#./install_ros.sh
