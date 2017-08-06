#!/bin/bash

sudo pip install -U \
     rosdep \
     rosinstall_generator \
     wstool \
     rosinstall

# Update ssl certs and init ros
sudo c_rehash /etc/ssl/certs
sudo rosdep init
rosdep update

# Initialize workspace. Could replace ros_comm with desktop or desktop_full
mkdir ~/ros_catkin_ws
cd ~/ros_catkin_ws
rosinstall_generator ros_comm --rosdistro lunar --deps --tar > lunar-ros_comm.rosinstall
wstool init -j$(nproc) src lunar-ros_comm.rosinstall
wstool update -j$(nproc) -t src # restart just in case

# resolve dependencies
rosdep install --from-paths src --ignore-src --rosdistro lunar -y

# Build workspace
./src/catkin/bin/catkin_make_isolated --install -DCMAKE_BUILD_TYPE=Release
