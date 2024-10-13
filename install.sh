#!/bin/bash
set -e

sudo apt-get update
sudo apt-get install -y ninja-build libgflags-dev libgtest-dev

INSTALL_PREFIX=/usr/local ./build.sh libraft template --compile-static-lib --compile-lib
