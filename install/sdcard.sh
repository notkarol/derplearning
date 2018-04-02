#!/bin/bash

# Variables that may be changed to alter the function of this script on other platforms
# Assumes existing disk is already partitioned to ext4
dev=/dev/mmcblk1p1
root=/mnt/sdcard

# Create a folder for the SD card
sudo mkdir -p $root

# Make sure that it's in fstab so it auto loads
echo "$dev $root ext4 errors=remount-ro 0 2" | sudo tee -a /etc/fstab

# Mount it so we can create base folders
sudo mount $root

# Make sure that this sd card is owned by the user who ran this
sudo chown -R ${USER}:${USER} $root

# Prepare derprc
cat > ~/.derprc <<EOF
export DERP_CODE=${root}/derplearning/src
export DERP_CONFIG=${root}/derplearning/src/config
export DERP_DATA=${root}/data
export DERP_MODEL=${root}/models
export DERP_SCRATCH=${root}/scratch
EOF
echo "source ~/.derprc" >> ~/.bashrc

# Prepare folders and code on sdcard
mkdir ${root}/data ${root}/models ${root}/scratch
ln -s $(dirname $PWD) ${root}/derplearning
