#!/bin/bash

# Make sure we're working off of latest packages
sudo apt update
sudo apt upgrade -y
sudo apt install \
     cmake \
     libatlas-base-dev \
     libbluetooth-dev \
     libffi-dev \
     libusb-1.0-0-dev \
     python3-dev \
     python3-matplotlib \
     python3-numpy \
     python3-pillow \
     python3-pip \
     python3-seaborn \
     python3-scipy \
     python3-zmq

# Install Python Packages
pip3 install --user --upgrade Adafruit-BNO055 evdev pybluez pyserial pyusb

# Enable bluetooth
sudo rfkill unblock bluetooth
sudo systemctl enable bluetooth
sudo systemctl start bluetooth

# Make sure we can communicate to Micro Maestro
sudo cp 99-pololu.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules

# Add groups we might need to use. netdev for bluetooth and i2c for gpio ports
if [[ -e $(groups | grep i2c) ]] ; then 
    sudo usermod -a -G i2c $USER # gpio
fi
if [[ -e $(groups | grep input) ]] ; then 
    sudo usermod -a -G input $USER # input devices
fi
if [[ -e $(groups | grep netdev) ]] ; then 
    sudo usermod -a -G netdev $USER # bluetooth
fi

# Setup SD card if we're a car otherwise just the derp variables
if [[ $(uname -m) -eq "aarch64" ]] ; then
    bash sdcard.sh
    echo "Please add the following line to 'crontab -e'"
    echo "* * * * * /bin/bash /mnt/sdcard/derplearning/dameon_ds4.sh"
else
    echo "export DERP_ROOT=${PWD}" >> ~/.derprc
    echo "source ~/.derprc" >> ~/.bashrc
fi
source ~/.derprc

# Compile and install other useful packages
bash install_opencv.sh
bash install_pycuda.sh
bash install_pytorch.sh
