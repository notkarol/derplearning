#!/bin/bash

# Variables that may be changed to alter the function of this script on other platforms
# Assumes existing disk is already partitioned to ext4
dev=/dev/mmcblk1p1
root=/dev/sdcard
folders=scratch data model

# Create a folder for the SD card
sudo mkdir -p $root

# Make sure that it's in fstab so it auto loads
echo "$dev $root ext4 errots=remount-ro 0 2" | sudo tee -a /etc/fstab

# Mount it so we can create base folders
sudo mount $root

# Create each folder that we wish to store on the sdcard
# Also add an entry in bashrc for an environment variable pointing to it
for folder in $folders
do
    sudo mkdir -p ${root}/${folder}
    upper=$(echo $folder | awk '{print toupper($0)}')
    echo "export DERP_${upper}=${root}/${folder}" >> $HOME/.bashrc
done

# Finally make sure that this sd card is owned by the user who ran this
sudo chown -R ${USER}:${USER} $root
