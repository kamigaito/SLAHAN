#!/bin/bash -eu

ROOT_DIR=${1}
BERT_BASE_DIR="${ROOT_DIR}/pretrained"
BERT_MODEL_NAME="cased_L-12_H-768_A-12"

# Extract  contextualized features
cd ${ROOT_DIR}/bert
python extract_features.py \
    --input_file=${ROOT_DIR}/dataset2/conv/test.cln.strip.sent \
    --output_file=${ROOT_DIR}/dataset2/conv/test.cln.strip.sent.bert.hdf5 \
    --vocab_file=${BERT_BASE_DIR}/${BERT_MODEL_NAME}/vocab.txt \
    --bert_config_file=${BERT_BASE_DIR}/${BERT_MODEL_NAME}/bert_config.json \
    --init_checkpoint=${BERT_BASE_DIR}/${BERT_MODEL_NAME}/bert_model.ckpt \
    --layers=-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12 \
    --max_seq_length=400 \
    --word_embeddings=True \
    --word_composition=avg \
    --batch_size=16
cd ${ROOT_DIR}
