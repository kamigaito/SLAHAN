#!/bin/bash -eu

ROOT_DIR=${1}
SCRIPTS_DIR=${ROOT_DIR}/scripts/setup/google

# Download resources
bash ${SCRIPTS_DIR}/download.sh ${ROOT_DIR} || exit

# Construct dataset
bash ${SCRIPTS_DIR}/convert.sh ${ROOT_DIR} || exit

# Extract features
bash ${SCRIPTS_DIR}/extract_features.sh ${ROOT_DIR} || exit
