#!/bin/bash -eu

ROOTDIR=${PWD}
SCRIPTSDIR=${ROOTDIR}/scripts/setup

# Make directories

if [ ! -d ${ROOTDIR}/dataset ]; then
    mkdir ${ROOTDIR}/dataset
fi

if [ ! -d ${ROOTDIR}/results ]; then
    mkdir ${ROOTDIR}/results
fi

# Compile

bash ${SCRIPTSDIR}/compile.sh ${ROOTDIR} || exit

# Download

bash ${SCRIPTSDIR}/download.sh ${ROOTDIR} || exit

# Setup for each dataset

## Google
bash ${SCRIPTSDIR}/google/setup.sh ${ROOTDIR} || exit
## BNC
bash ${SCRIPTSDIR}/bcn/setup.sh ${ROOTDIR} || exit
