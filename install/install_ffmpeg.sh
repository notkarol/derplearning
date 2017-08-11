#!/bin/bash

sudo apt-get update
sudo apt-get -y install \
     autoconf \
     automake \
     build-essential \
     libass-dev \
     libfreetype6-dev \
     libx264-dev \
     libx265-dev \
     libsdl2-dev \
     libtheora-dev \
     libtool \
     libva-dev \
     libvdpau-dev \
     libvorbis-dev \
     libxcb1-dev \
     libxcb-shm0-dev \
     libxcb-xfixes0-dev \
     pkg-config \
     texinfo \
     wget \
     yasm \
     zlib1g-dev

mkdir ~/ffmpeg_sources
cd ~/ffmpeg_sources
wget http://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2
tar xjvf ffmpeg-snapshot.tar.bz2
cd ffmpeg
PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
    --prefix="$HOME/ffmpeg_build" \
    --pkg-config-flags="--static" \
    --extra-cflags="-I$HOME/ffmpeg_build/include" \
    --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
    --bindir="$HOME/bin" \
    --enable-gpl \
    --enable-libass \
    --enable-libfreetype \
    --enable-libtheora \
    --enable-libvorbis \
    --enable-libx264 \
    --enable-libx265
PATH="$HOME/bin:$PATH" make
make install
hash -r

echo 'export PATH=${PATH}:${HOME}/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/lib' >> ~/.bashrc
rm -rf ~/ffmpeg-sources
