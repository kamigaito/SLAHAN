#!/bin/bash -u

DATADIR=${PWD}/dataset/conv
MODELDIR=${PWD}/models
SCRIPTSDIR=${PWD}/scripts/tools
RESULTSDIR=${PWD}/results
XMLDIR=/tmp

# models

NAMES=("lstm")
ROOTDIRS=("${MODELDIR}/lstm")
NAMES=("${NAMES[@]}" "lstm-dep")
ROOTDIRS=("${ROOTDIRS[@]}" "${MODELDIR}/lstm-dep")
NAMES=("${NAMES[@]}" "attn")
ROOTDIRS=("${ROOTDIRS[@]}" "${MODELDIR}/attn")
NAMES=("${NAMES[@]}" "tagger")
ROOTDIRS=("${ROOTDIRS[@]}" "${MODELDIR}/tagger")
NAMES=("${NAMES[@]}" "base")
ROOTDIRS=("${ROOTDIRS[@]}" "${MODELDIR}/base")
NAMES=("${NAMES[@]}" "slahan_w_syn")
ROOTDIRS=("${ROOTDIRS[@]}" "${MODELDIR}/slahan_w_syn")
NAMES=("${NAMES[@]}" "parent_w_syn")
ROOTDIRS=("${ROOTDIRS[@]}" "${MODELDIR}/parent_w_syn")
NAMES=("${NAMES[@]}" "child_w_syn")
ROOTDIRS=("${ROOTDIRS[@]}" "${MODELDIR}/child_w_syn")
NAMES=("${NAMES[@]}" "slahan_wo_syn")
ROOTDIRS=("${ROOTDIRS[@]}" "${MODELDIR}/slahan_wo_syn")
NAMES=("${NAMES[@]}" "parent_wo_syn")
ROOTDIRS=("${ROOTDIRS[@]}" "${MODELDIR}/parent_wo_syn")
NAMES=("${NAMES[@]}" "child_wo_syn")
ROOTDIRS=("${ROOTDIRS[@]}" "${MODELDIR}/child_wo_syn")

# For all sentences

echo "Model,1,2,3,AVG" > ${RESULTSDIR}/test_google_char_cr.csv

# gold compression ratio

python ${SCRIPTSDIR}/conv2text.py \
    -l ${DATADIR}/test.cln.label \
    -t ${DATADIR}/test.cln.sent \
> ${DATADIR}/test.cln.gold

echo -n "gold" >> ${RESULTSDIR}/test_google_char_cr.csv
for cnt in `seq 1 3`; do
    CCR=`python ${SCRIPTSDIR}/eval_char_len.py -i ${DATADIR}/test.cln.strip.sent -s ${DATADIR}/test.cln.gold |grep 'Compression Ratio '|awk -F" " '{print $4}'`
    echo -n ","${CCR} >> ${RESULTSDIR}/test_google_char_cr.csv
done
test_ccr=$(tail -n 1 ${RESULTSDIR}/test_google_char_cr.csv |awk -F"," '{print ($2+$3+$4) / 3}')
echo ",${test_ccr}" >> ${RESULTSDIR}/test_google_char_cr.csv

# system compression ratio
i=0
for name in "${NAMES[@]}"; do
    NAME=${NAMES[$i]}
    echo ${NAME}
    for model_id in `seq 0 2`; do
        ROOTDIR=${ROOTDIRS[$i]}_${model_id}
        python ${SCRIPTSDIR}/conv2text.py -t ${DATADIR}/test.cln.sent -l ${ROOTDIR}/test_result_greedy.sents > ${ROOTDIR}/test.comp &
    done
    wait
    echo -n ${NAME} >> ${RESULTSDIR}/test_google_char_cr.csv
    for model_id in `seq 0 2`; do
        ROOTDIR=${ROOTDIRS[$i]}_${model_id}
        CCR=`python ${SCRIPTSDIR}/eval_char_len.py -i ${DATADIR}/test.cln.strip.sent -s ${ROOTDIR}/test.comp |grep 'Compression Ratio '|awk -F" " '{print $4}'`
        echo -n ","${CCR} >> ${RESULTSDIR}/test_google_char_cr.csv
    done
    test_ccr=$(tail -n 1 ${RESULTSDIR}/test_google_char_cr.csv |awk -F"," '{print ($2+$3+$4) / 3}')
    echo ",${test_ccr}" >> ${RESULTSDIR}/test_google_char_cr.csv

    let i++

done

# For all sentences

echo "Model,1,2,3,AVG" > ${RESULTSDIR}/test_google_long_char_cr.csv

# gold compression ratio

python ${SCRIPTSDIR}/conv2text.py \
    -l ${DATADIR}/test.cln.label \
    -t ${DATADIR}/test.cln.sent \
> ${DATADIR}/test.cln.gold

echo -n "gold" >> ${RESULTSDIR}/test_google_long_char_cr.csv
for cnt in `seq 1 3`; do
    CCR=`python ${SCRIPTSDIR}/eval_char_len.py -l 30 -i ${DATADIR}/test.cln.strip.sent -s ${DATADIR}/test.cln.gold |grep 'Compression Ratio '|awk -F" " '{print $4}'`
    echo -n ","${CCR} >> ${RESULTSDIR}/test_google_long_char_cr.csv
done
test_ccr=$(tail -n 1 ${RESULTSDIR}/test_google_long_char_cr.csv |awk -F"," '{print ($2+$3+$4) / 3}')
echo ",${test_ccr}" >> ${RESULTSDIR}/test_google_long_char_cr.csv

# system compression ratio
i=0
for name in "${NAMES[@]}"; do
    NAME=${NAMES[$i]}
    echo ${NAME}
    for model_id in `seq 0 2`; do
        ROOTDIR=${ROOTDIRS[$i]}_${model_id}
        python ${SCRIPTSDIR}/conv2text.py -t ${DATADIR}/test.cln.sent -l ${ROOTDIR}/test_result_greedy.sents > ${ROOTDIR}/test.comp &
    done
    wait
    echo -n ${NAME} >> ${RESULTSDIR}/test_google_long_char_cr.csv
    for model_id in `seq 0 2`; do
        ROOTDIR=${ROOTDIRS[$i]}_${model_id}
        CCR=`python ${SCRIPTSDIR}/eval_char_len.py -l 30 -i ${DATADIR}/test.cln.strip.sent -s ${ROOTDIR}/test.comp |grep 'Compression Ratio '|awk -F" " '{print $4}'`
        echo -n ","${CCR} >> ${RESULTSDIR}/test_google_long_char_cr.csv
    done
    test_ccr=$(tail -n 1 ${RESULTSDIR}/test_google_long_char_cr.csv |awk -F"," '{print ($2+$3+$4) / 3}')
    echo ",${test_ccr}" >> ${RESULTSDIR}/test_google_long_char_cr.csv

    let i++

done

