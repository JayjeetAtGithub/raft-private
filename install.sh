#!/bin/bash
set -e

INSTALL_PREFIX=/usr/local ./build.sh libraft template --compile-static-lib
