#!/bin/bash -eu

ROOT_DIR=${1}

# Extract  contextualized features
cd ${ROOT_DIR}
allennlp elmo \
    ${ROOT_DIR}/dataset2/conv/test.cln.strip.sent \
    ${ROOT_DIR}/dataset2/conv/test.cln.strip.sent.elmo.hdf5 \
    --cuda-device 0 \
    --all
