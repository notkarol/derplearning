#!/bin/bash

# update hostname
echo "Please enter the hostname of this machine"
read hostname
sudo sed -i "s/raspberrypi/$hostname" /etc/hostname
sudo sed -i "s/raspberrypi/$hostname" /etc/hosts

# update password
passwd

# Configure a lot of things
sudo dpkg-reconfigure locales
sudo dpkg-reconfigure keyboard-configuration
sudo dpkg-reconfigure openssh-server

# Prepare packages
sudo apt update
sudo apt upgrade -y
sudo apt install \
     git

# Install emacs and useful related packages
sudo apt install \
     emacs25-nox \
     python-mode
echo "(load-theme 'tsdh-dark)" >> ~/.emacs


# Get the repository
git clone https://github.com/John-Ellis/derplearning

cd derplearning/install
./install.sh
