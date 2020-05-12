#!/bin/bash -eu

ROOT_DIR=${1}
IN_DIR="${ROOT_DIR}/dataset/conv"
OUT_DIR="${ROOT_DIR}/dataset/conv"
PREFIX_LIST=("dev.cln" "test.cln")

# Extract  contextualized features
cd ${ROOT_DIR}
for prefix in ${PREFIX_LIST[@]}; do
    allennlp elmo \
    ${IN_DIR}/${prefix}.strip.sent \
    ${OUT_DIR}/${prefix}.strip.sent.elmo.hdf5 \
    --cuda-device 0 \
    --all
done
