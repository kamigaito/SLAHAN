#!/bin/bash -eu

# Settings
MODEL_ID=${1}
DATA_DIR=${PWD}/dataset/conv
MODEL_DIR=${PWD}/models
DIM_SIZE="200"
LAYER_SIZE="2"
HDF5_FILES_TRAIN="${DATADIR}/train.cln.strip.sent.bert.hdf5,${DATADIR}/train.cln.strip.sent.elmo.hdf5,${DATADIR}/train.cln.strip.sent.glove.hdf5"
HDF5_FILES_DEV="${DATADIR}/dev.strip.sent.bert.hdf5,${DATADIR}/dev.strip.sent.elmo.hdf5,${DATADIR}/dev.strip.sent.glove.hdf5"
HDF5_DIMS="768,1024,300"
HDF5_LAYERS="12,3,1"
ALIGN_WEIGHT=1.0
REC_ATTN="1,2,3,4"
CTX_TYPE="child"
SELECTIVE_GATE="1"

# Prepare a model root directory
ROOT_DIR=${MODEL_DIR}/child_w_syn_${MODEL_ID}
if [ ! -d ${ROOT_DIR} ]; then
    mkdir -p ${ROOT_DIR}
fi

# Run
${PWD}/build_compressor/slahan \
--mode train \
--rootdir ${ROOT_DIR} \
--srcfile ${DATA_DIR}/train-large.cln.sent \
--trgfile ${DATA_DIR}/train-large.cln.label \
--alignfile ${DATA_DIR}/train-large.cln.dep \
--srcvalfile ${DATA_DIR}/dev.cln.sent \
--trgvalfile ${DATA_DIR}/dev.cln.label \
--alignvalfile ${DATA_DIR}/dev.cln.dep \
--elmo_hdf5_files ${HDF5_FILES_TRAIN} \
--elmo_hdf5_dev_files ${HDF5_FILES_DEV} \
--elmo_hdf5_dims ${HDF5_DIMS} \
--elmo_hdf5_layers ${HDF5_LAYERS} \
--lookup_type "none" \
--enc_char_vocab_size -1 \
--enc_char_vec_size 1 \
--enc_word_vocab_size 1 \
--enc_word_vec_size 1 \
--dec_word_vocab_size -1 \
--dec_word_vec_size 0 \
--char_num_layers 1 \
--char_rnn_size 1 \
--num_layers ${LAYER_SIZE} \
--rnn_size ${DIM_SIZE} \
--att_size ${DIM_SIZE} \
--additional_connect_layer 0 \
--additional_output_layer 1 \
--guided_alignment 1 \
--guided_alignment_weight ${ALIGN_WEIGHT} \
--guided_alignment_decay 1.0 \
--guided_alignment_start_epoch 1 \
--guided_alignment_decay_for_each 1 \
--self_attn_type "self" \
--rec_attn_dep ${REC_ATTN} \
--selective_gate ${SELECTIVE_GATE} \
--context_type ${CTX_TYPE} \
--optim adam \
--learning_rate 0.001 \
--lr_auto_decay 0 \
--lr_decay 1.0 \
--start_epoch 5 \
--decay_for_each 5 \
--sgd_start_epoch 10000 \
--sgd_start_decay 10 \
--sgd_start_decay_for_each 1 \
--sgd_start_learning_rate 0.01 \
--sgd_start_lr_decay 0.9 \
--clipping_enabled 1 \
--clip_threshold 5.0 \
--drop_word_alpha 0.0 \
--dynet-weight-decay 0.0 \
--max_batch_train 16 \
--max_batch_pred 5 \
--sort_sent_type_train "sort_random" \
--batch_type_train "default" \
--shuffle_batch_type_train "random" \
--src_tok_lim_train 2000 \
--trg_tok_lim_train 2000 \
--sort_sent_type_pred "sort_random" \
--batch_type_pred "same_length" \
--shuffle_batch_type_pred "default" \
--src_tok_lim_pred 2000 \
--trg_tok_lim_pred 2000 \
--decoder_type "greedy" \
--dropout_rate_lstm_char 0.0 \
--dropout_rate_lstm_word 0.3 \
--max_length 500 \
--epochs 20
