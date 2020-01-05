#!/bin/bash -x

# Install necessary distribution packages
sudo apt update
sudo apt install -y \
     git \
     cmake \
     ffmpeg \
     gfortran \
     libatlas-base-dev \
     libavcodec-dev \
     libavformat-dev \
     libbluetooth-dev \
     libffi-dev \
     libgstreamer1.0-dev \
     libgstreamer-plugins-base1.0-dev \
     libgtk2.0-dev \
     libjpeg-dev \
     libswscale-dev \
     libusb-1.0-0-dev \
     pkg-config \
     python3-dev \
     python3-evdev \
     python3-matplotlib \
     python3-pillow \
     python3-pip \
     python3-skimage \
     python3-pyudev \
     python3-zmq \
     zlib1g-dev

# Install python packages one at a time to ensure it works
for package in cython pycapnp numpy scipy==1.1.0 PyYAML Adafruit-BNO055 pybluez pyserial pyusb ; do
    pip3 install --user $package
done

# Install Pytorch, separate platforms require separate solutions
if [[ "$(uname -m)" == "x86_64" ]] ; then
    pip3 install --user torch torchvision
elif [[ -n $(egrep -i "jetson.nano" /proc/device-tree/model) ]] ; then
    if [[ -z $(pip3 freeze | grep torch) ]] ; then
	wget https://nvidia.box.com/shared/static/phqe92v26cbhqjohwtvxorrwnmrnfx1o.whl -O torch-1.3.0-cp36-cp36m-linux_aarch64.whl
	pip3 install --user torch-1.3.0-cp36-cp36m-linux_aarch64.whl torchvision
    fi
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
hidraw_path=/etc/udev/rules.d/99-hidraw-permissions.rules
if ! [[ -e $hidraw_path ]] ; then
    echo 'KERNEL=="hidraw*", SUBSYSTEM=="hidraw", MODE="0660", GROUP="plugdev"' | sudo tee $hidraw_path
    sudo udevadm control --reload-rules
fi
if [[ -e $(groups | grep i2c) ]] ; then
    sudo usermod -a -G i2c $USER # gpio
fi
if [[ -e $(groups | grep netdev) ]] ; then
    sudo usermod -a -G netdev $USER # bluetooth
fi
if [[ -e $(groups | grep input) ]] ; then
    sudo usermod -a -G input $USER # evdev input
fi

# Set up our script with sourcing instructions
if ! [[ -e $PWD/recordings ]] ; then
    mkdir -p $PWD/recordings $PWD/models
    echo "export DERP_ROOT=$PWD" >> ~/.bashrc
    echo "export PYTHONPATH=$PYTHONPATH:$DERP_ROOT/capnp" >> ~/.bashrc
    source ~/.bashrc
fi

# Install the derp python package
python3 setup.py install --user

echo "Please reboot your machine"
