#!/bin/bash -eu

ROOT_DIR=${1}
PRETRAIN_DIR="${ROOT_DIR}/pretrained"
BERT_MODEL_NAME="cased_L-12_H-768_A-12"
GLOVE_MODEL_NAME="glove.840B.300d"

if [ ! -e ${PRETRAIN_DIR} ]
then
    mkdir -p ${PRETRAIN_DIR}
fi

# Download pretrained models
cd ${PRETRAIN_DIR}

# Download a BERT model
wget --no-check-certificate https://storage.googleapis.com/bert_models/2018_10_18/${BERT_MODEL_NAME}.zip
unzip ${BERT_MODEL_NAME}.zip

# Download GloVe
wget --no-check-certificate http://nlp.stanford.edu/data/${GLOVE_MODEL_NAME}.zip
unzip ${GLOVE_MODEL_NAME}.zip

# Download stanford dependency parser
cd ${ROOT_DIR}
wget --no-check-certificate https://nlp.stanford.edu/software/stanford-parser-full-2018-10-17.zip
unzip stanford-parser-full-2018-10-17.zip

# Download trained models
FILE_ID=1HGeJsevuklHr2zbG3SgGKnuG7wXFq_18
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o models.zip
unzip models.zip
