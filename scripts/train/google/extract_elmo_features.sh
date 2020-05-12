#!/bin/bash -eu

ROOT_DIR=${1}
IN_DIR="${ROOT_DIR}/dataset/conv"
OUT_DIR="${ROOT_DIR}/dataset/conv"

# Extract  contextualized features

cd ${ROOT_DIR}

allennlp elmo \
    ${IN_DIR}/train-large.cln.strip.sent \
    ${OUT_DIR}/train-large.cln.strip.sent.elmo.hdf5 \
    --cuda-device 0 \
    --all
