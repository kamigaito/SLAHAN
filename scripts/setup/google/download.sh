#!/bin/bash -eu

ROOT_DIR=${1}
IN_DIR="${ROOT_DIR}/dataset/orig"

if [ -d ${IN_DIR} ]
then
    rm -rf ${IN_DIR}
fi
mkdir -p ${IN_DIR}

# Download google sentence compression dataset
git clone https://github.com/google-research-datasets/sentence-compression.git ${IN_DIR}
