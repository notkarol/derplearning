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

mkdir ffmpeg_sources
cd ffmpeg_sources
wget http://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2
tar xjvf ffmpeg-snapshot.tar.bz2
cd ffmpeg
PATH="/usr/local/bin:$PATH" PKG_CONFIG_PATH="${PWD}/ffmpeg_build/lib/pkgconfig" ./configure \
    --prefix="${PWD}/ffmpeg_build" \
    --pkg-config-flags="--static" \
    --extra-cflags="-I${PWD}/ffmpeg_build/include" \
    --extra-ldflags="-L${PWD}/ffmpeg_build/lib" \
    --bindir="/usr/local/bin" \
    --enable-gpl \
    --enable-libass \
    --enable-libfreetype \
    --enable-libtheora \
    --enable-libvorbis \
    --enable-libx264 \
    --enable-libx265
PATH="/usr/local/bin:$PATH" make
sudo make install
hash -r

# Cleanup
cd ../..
rm -rf ~/ffmpeg-sources
