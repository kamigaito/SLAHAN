#!/bin/bash -eu

SCRIPTS_DIR=${PWD}/scripts/tools
CORPUS_MAX_LENGTH=200
INDIR=${PWD}/dataset/orig
OUTDIR=${PWD}/dataset/conv

if [ ! -d ${OUTDIR} ]; then
    mkdir -p ${OUTDIR}
fi

# Convert original dataset

python ${SCRIPTS_DIR}/convert_google_json.py \
--input-json-file <(gzip -dc ${INDIR}/data/comp-data.eval.json.gz) \
--sent-file ${OUTDIR}/comp-data.eval.sent \
--label-file ${OUTDIR}/comp-data.eval.label \
--dep-file ${OUTDIR}/comp-data.eval.dep \
--rel-file ${OUTDIR}/comp-data.eval.rel \
--pos-file ${OUTDIR}/comp-data.eval.pos

suffix_list=("sent" "label" "dep" "rel" "pos")

# Test

for suffix in ${suffix_list[@]}; do
    cat ${OUTDIR}/comp-data.eval.${suffix} |\
    head -n 1000 \
    > ${OUTDIR}/test.${suffix}
done

python ${SCRIPTS_DIR}/clean_corpus.py \
-i ${OUTDIR}/test \
-o ${OUTDIR}/test.cln \
-m 0

# Train

for suffix in ${suffix_list[@]}; do
    cat ${OUTDIR}/comp-data.eval.${suffix} |\
    tail -n 9000 |
    head -n 8000 \
    > ${OUTDIR}/train.${suffix}
done

python ${SCRIPTS_DIR}/clean_corpus.py \
-i ${OUTDIR}/train \
-o ${OUTDIR}/train.cln \
-m ${CORPUS_MAX_LENGTH}

# Dev

for suffix in ${suffix_list[@]}; do
    cat ${OUTDIR}/comp-data.eval.${suffix} |\
    tail -n 1000 \
    > ${OUTDIR}/dev.${suffix}
done

python ${SCRIPTS_DIR}/clean_corpus.py \
-i ${OUTDIR}/dev \
-o ${OUTDIR}/dev.cln \
-m 0

# Concatenation

prefix_list=("train" "dev" "test")

for prefix in ${prefix_list[@]}; do
    python ${SCRIPTS_DIR}/feature.py \
        --sent-file ${OUTDIR}/${prefix}.sent \
        --pos-file ${OUTDIR}/${prefix}.pos \
        --rel-file ${OUTDIR}/${prefix}.rel \
        --out-file ${OUTDIR}/${prefix}.feat
done

# Large train

for i in `seq -f %02g 1 10`
do
    python ${SCRIPTS_DIR}/convert_google_json.py \
    --conv-large \
    --input-json-file <(gzip -dc ${INDIR}/data/sent-comp.train${i}.json.gz) \
    --sent-file ${OUTDIR}/train${i}.sent \
    --label-file ${OUTDIR}/train${i}.label \
    --dep-file ${OUTDIR}/train${i}.dep \
    --rel-file ${OUTDIR}/train${i}.rel \
    --pos-file ${OUTDIR}/train${i}.pos
done

echo -n > ${OUTDIR}/train-large.sent
echo -n > ${OUTDIR}/train-large.label
echo -n > ${OUTDIR}/train-large.dep
echo -n > ${OUTDIR}/train-large.rel
echo -n > ${OUTDIR}/train-large.pos

for i in `seq -f %02g 1 10`
do
    cat ${OUTDIR}/train${i}.sent >> ${OUTDIR}/train-large.sent
    cat ${OUTDIR}/train${i}.label >> ${OUTDIR}/train-large.label
    cat ${OUTDIR}/train${i}.dep >> ${OUTDIR}/train-large.dep
    cat ${OUTDIR}/train${i}.rel >> ${OUTDIR}/train-large.rel
    cat ${OUTDIR}/train${i}.pos >> ${OUTDIR}/train-large.pos
done

python ${SCRIPTS_DIR}/clean_corpus.py \
-i ${OUTDIR}/train-large \
-o ${OUTDIR}/train-large.cln \
-m ${CORPUS_MAX_LENGTH}

python ${SCRIPTS_DIR}/feature.py \
    --sent-file ${OUTDIR}/train-large.cln.sent \
    --pos-file ${OUTDIR}/train-large.cln.pos \
    --rel-file ${OUTDIR}/train-large.cln.rel \
    --out-file ${OUTDIR}/train-large.cln.feat

# Postprocess
PREFIX_LIST=("train-large.cln" "train.cln" "dev.cln" "test.cln")

for prefix in ${PREFIX_LIST[@]}; do
    # Strip start and end of sentence symbols
    cat ./dataset/conv/${prefix}.sent |\
    sed 's/^<s> //g' |\
    sed 's/ <\/s>$//g' \
    > ./dataset/conv/${prefix}.strip.sent
    # Strip start and end of sentence symbols
    cat ./dataset/conv/${prefix}.label |\
    sed 's/^<s> //g' |\
    sed 's/ <\/s>$//g' \
    > ./dataset/conv/${prefix}.strip.label
done
