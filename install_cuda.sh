#!/bin/bash
set -vex
uname -a
cat /proc/version

apt-get -qq update
apt-get -yqq install wget libxml2-dev
wget -q https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
sh cuda_11.6.0_510.39.01_linux.run --help
sh cuda_11.6.0_510.39.01_linux.run --silent --toolkit
cat /var/log/cuda-installer.log

ls /usr/local/cuda
