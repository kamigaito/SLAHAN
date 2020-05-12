#!/bin/bash -eu

ROOTDIR=${PWD}
SCRIPTSDIR=${ROOTDIR}/scripts/train/google

bash ${SCRIPTSDIR}/extract_bert_features.sh ${ROOTDIR} || exit
bash ${SCRIPTSDIR}/extract_elmo_features.sh ${ROOTDIR} || exit
bash ${SCRIPTSDIR}/extract_glove_features.sh ${ROOTDIR} || exit
