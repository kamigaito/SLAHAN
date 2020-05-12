#!/bin/bash -eu

ROOT_DIR=${1}

cd ${ROOT_DIR}

python ${ROOT_DIR}/scripts/tools/conll2dep.py \
-i ${ROOT_DIR}/dataset2/conv/test.cln.strip.dep.conll \
-o ${ROOT_DIR}/dataset2/conv/test.cln.dep
