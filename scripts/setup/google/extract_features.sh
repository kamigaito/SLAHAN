#!/bin/bash -eu

ROOTDIR=${1}
SCRIPTSDIR=${ROOTDIR}/scripts/setup

bash ${SCRIPTSDIR}/google/extract_bert_features.sh ${ROOTDIR} || exit
bash ${SCRIPTSDIR}/google/extract_elmo_features.sh ${ROOTDIR} || exit
bash ${SCRIPTSDIR}/google/extract_glove_features.sh ${ROOTDIR} || exit
