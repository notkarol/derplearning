#!/bin/bash

sudo apt update
sudo apt upgrade -y

./install_opencv3.sh
./install_tensorflow.sh
./install_pytorch.sh
./install_ros.sh
