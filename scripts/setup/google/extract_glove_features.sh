#!/bin/bash -eu

ROOT_DIR=${1}
GLOVE_BASE_DIR="${ROOT_DIR}/pretrained"
GLOVE_MODEL_NAME="glove.840B.300d.txt"
PREFIX_LIST=("dev.cln" "test.cln")

# Extract glove features
cd ${ROOT_DIR}
for prefix in ${PREFIX_LIST[@]}; do
    python ${ROOT_DIR}/scripts/tools/extract_glove_features.py \
    --input_file=${ROOT_DIR}/dataset/conv/${prefix}.strip.sent \
    --output_file=${ROOT_DIR}/dataset/conv/${prefix}.strip.sent.glove.hdf5 \
    --model_file=${GLOVE_BASE_DIR}/${GLOVE_MODEL_NAME}
done
