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
pip3 install --user --upgrade Adafruit-BNO055
pip3 install --user --upgrade evdev
pip3 install --user --upgrade pybluez
pip3 install --user --upgrade pyserial
pip3 install --user --upgrade pyusb

# Enable bluetooth
rfkill unblock bluetooth
systemctl enable bluetooth
systemctl start bluetooth

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
    echo '* * * * * /bin/bash -x /mnt/sdcard/derplearning/src/ds4daemon.sh'
else
    cat > ~/.derprc <<EOF
export DERP_CODE=$(dirname $PWD)/src
export DERP_CONFIG=$(dirname $PWD)/src/config
export DERP_DATA=${HOME}/data
export DERP_MODEL=${HOME}/models
export DERP_SCRATCH=${HOME}/scratch
EOF
    echo "source ~/.derprc" >> ~/.bashrc
fi
source ~/.derprc

# Install v4l2capture, a python&C library for interfacing with cameras
bash v4l2capture.sh

# Compile OpenCV from source
bash opencv.sh

# Compile PyCuda from source
bash pycuda.sh

# Compile PyTorch from source
bash pytorch.sh

