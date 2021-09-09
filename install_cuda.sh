#!/bin/bash
set -vex
uname -a
cat /proc/version

apt-get -qq update
apt-get -yqq install wget libxml2-dev
wget -q https://developer.download.nvidia.com/compute/cuda/11.4.2/local_installers/cuda_11.4.2_470.57.02_linux.run
sh cuda_11.4.2_470.57.02_linux.run --help
sh cuda_11.4.2_470.57.02_linux.run --silent --toolkit
cat /var/log/cuda-installer.log

#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
#mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
#wget -q https://developer.download.nvidia.com/compute/cuda/11.4.2/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.2-470.57.02-1_amd64.deb
#dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.2-470.57.02-1_amd64.deb
#APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1 apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
#apt-get update
#apt-get -yqq install cuda
#apt-get update

ls /usr/local/cuda
