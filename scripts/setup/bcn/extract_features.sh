#!/bin/bash -eu

ROOTDIR=${1}
SCRIPTSDIR=${ROOTDIR}/scripts/setup

bash ${SCRIPTSDIR}/bcn/extract_bert_features.sh ${ROOTDIR}
bash ${SCRIPTSDIR}/bcn/extract_elmo_features.sh ${ROOTDIR}
bash ${SCRIPTSDIR}/bcn/extract_glove_features.sh ${ROOTDIR}
