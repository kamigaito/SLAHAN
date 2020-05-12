#!/bin/bash -eu

ROOT_DIR=${1}
GLOVE_BASE_DIR="${ROOT_DIR}/pretrained"
GLOVE_MODEL_NAME="glove.840B.300d.txt"

# Extract glove features

cd ${ROOT_DIR}

python ${ROOT_DIR}/scripts/tools/extract_glove_features.py \
    --input_file=${ROOT_DIR}/dataset/conv/train-large.cln.strip.sent \
    --output_file=${ROOT_DIR}/dataset/conv/train-large.cln.strip.sent.glove.hdf5 \
    --model_file=${GLOVE_BASE_DIR}/${GLOVE_MODEL_NAME}
