#!/bin/bash -x

DERP_CONFIG=$PWD/config
DERP_BRAIN=$HOME/brain
DERP_RECORDING=$HOME/recording

# Make sure we're working off of latest packages
sudo apt update
sudo apt upgrade -y
sudo apt install -y \
     git \
     cmake \
     ffmpeg \
     libatlas-base-dev \
     libbluetooth-dev \
     libffi-dev \
     libjpeg-dev \
     libusb-1.0-0-dev \
     python3-dev \
     python3-matplotlib \
     python3-numpy \
     python3-pillow \
     python3-pip \
     python3-seaborn \
     python3-scipy \
     python3-zmq \
     zlib1g-dev

# Install Python Packages
if [[ -z $(pip3 freeze | grep torch) ]] ; then
    wget https://nvidia.box.com/shared/static/06vlvedmqpqstu1dym49fo7aapgfyyu9.whl -O torch-1.2.0a0+8554416-cp36-cp36m-linux_aarch64.whl
    pip3 install --user numpy torch-1.2.0a0+8554416-cp36-cp36m-linux_aarch64.whl

    # Install torchvision directly
    git clone -b v0.3.0 https://github.com/pytorch/vision torchvision
    cd torchvision
    python3 setup.py install --user
    cd ..
    rm -rf torchvision
fi
    #pip3 install --user torchvision
if [[ -z $(pip3 freeze | grep Adafruit-BNO055) ]] ; then
    pip3 install --user --upgrade Adafruit-BNO055 evdev pybluez pyserial pyusb
fi

# Enable bluetooth
sudo rfkill unblock bluetooth
sudo systemctl enable bluetooth
sudo systemctl start bluetooth

# Make sure we can communicate to Micro Maestro and other devices
maestro_path=/etc/udev/rules.d/99-pololu.rules
if ! [[ -e $maestro_path ]] ; then
    echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1ffb", MODE="0666"' | sudo tee $maestro_path
    sudo udevadm control --reload-rules
fi
if [[ -e $(groups | grep i2c) ]] ; then 
    sudo usermod -a -G i2c $USER # gpio
fi
if [[ -e $(groups | grep input) ]] ; then 
    sudo usermod -a -G input $USER # input devices
fi
if [[ -e $(groups | grep netdev) ]] ; then 
    sudo usermod -a -G netdev $USER # bluetooth
fi

# Set up our script with sourcing instructions
if ! [[ -e ~/.derprc ]] ; then 
    mkdir -p $HOME/models $HOME/recordings
    echo "export DERP_CONFIG=$DERP_CONFIG" >> ~/.derprc
    echo "export DERP_BRAIN=$DERP_BRAIN" >> ~/.derprc
    echo "export DERP_DATA=$DERP_DATA" >> ~/.derprc
    echo "source ~/.derprc" >> ~/.bashrc
    source ~/.derprc
fi

