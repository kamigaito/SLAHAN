#!/bin/bash -eu

ROOT_DIR=${1}
SCRIPTS_DIR=${ROOT_DIR}/scripts/setup/bcn

# Download resources
bash ${SCRIPTS_DIR}/download.sh ${ROOT_DIR} || exit

# Construct dataset
bash ${SCRIPTS_DIR}/convert.sh ${ROOT_DIR} || exit

# Parse sentences
bash ${SCRIPTS_DIR}/parsing.sh ${ROOT_DIR} || exit
bash ${SCRIPTS_DIR}/conll2dep.sh ${ROOT_DIR} || exit

# Extract features
bash ${SCRIPTS_DIR}/extract_features.sh ${ROOT_DIR} || exit
