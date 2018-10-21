#!/bin/bash

# Location where we should install our software
install_path=/mnt/sdcard
install_sdcard_device=/dev/mmcblk1p1

# Make sure we're working off of latest packages
sudo apt update
sudo apt upgrade -y
sudo apt install \
     git \
     cmake \
     ffmpeg \
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

# Setup SD card if we're a car otherwise it's the s
DERP_ROOT=${PWD}
if [[ $(uname -m) -eq "aarch64" ]] && [[ -e $install_sdcard_device ]] && ! [[ -e $install_path ]]; then

    # Assume existing disk is already partitioned to ext4. Create an fstab entry and mount it
    sudo mkdir -p $install_path
    echo "$install_sdcard_device $install_path ext4 auto,nofail,errors=remount-ro 0 0" | sudo tee -a /etc/fstab
    sudo mount $install_path
    sudo chown -R ${USER}:${USER} $install_path

    # Move our current folder onto the sdcard
    DERP_ROOT=${install_path}/derplearning
    cd ../..
    mv derplearning ${install_path}
    cd $DERP_ROOT

    # Prepare daemon for the background
    echo "* * * * * su $USER -c '/bin/bash $DERP_ROOT/dameon_ds4.sh'" | sudo tee -a /etc/cron.d/daemon_ds4
fi

# Set up our script with sourcing instructions
if ! [[ -e ~/.derprc ]] ; then 
    echo "export DERP_ROOT=${DERP_ROOT}" >> ~/.derprc
    echo "source ~/.derprc" >> ~/.bashrc
    source ~/.derprc
fi

# Prepare useful folders
mkdir -p ../models ../scratch ../data

# Compile and install other useful packages

bash install_opencv.sh
bash install_pytorch.sh
