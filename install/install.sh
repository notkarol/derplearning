#!/bin/bash

# Make sure we're working off of latest packages
sudo apt update
sudo apt upgrade -y
sudo apt install \
     libusb-1.0-0-dev \
     libbluetooth-dev \
     libffi-dev \
     libopenblas-dev

# Install Python Packages
pip3 install --user -U pyusb
pip3 install --user -U evdev
pip3 install --user -U pybluez
pip3 install --user -U pyserial
pip3 install --user -U Adafruit-BNO055

# Make sure we can communicate to Micro Maestro
sudo cp 99-pololu.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules

# Add groups we might need to use. netdev for bluetooth and i2c for gpio ports
sudo usermod -a -G i2c $USER # gpio
sudo usermod -a -G input $USER # input devices
sudo usermod -a -G netdev $USER # bluetooth

# Install v4l2capture, a python&C library for interfacing with cameras
bash v4l2capture.sh

# Compile OpenCV3 from source
bash opencv3.sh

# Compile PyTorch from source
bash pytorch.sh
