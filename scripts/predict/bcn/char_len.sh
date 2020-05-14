#!/bin/bash -u

DATADIR=./dataset2/conv
MODELDIR=./models
BINDIR=./build_compressor
SCRIPTSDIR=./scripts/tools
RESULTSDIR=./results
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

# All

echo "Model,1-1,1-2,1-3,2-1,2-2,2-3,3-1,3-2,3-3,AVG" > ${RESULTSDIR}/test_bcn_char_cr.csv

# gold compression ratio

echo -n "gold" >> ${RESULTSDIR}/test_bcn_char_cr.csv

for data_id in `seq 1 3`; do
    for cnt in `seq 1 3`; do
        CCR=`python ${SCRIPTSDIR}/eval_char_len.py -i ${DATADIR}/test.cln.strip.sent -s ${DATADIR}/test.cln.${data_id}.gold |grep 'Compression Ratio '|awk -F" " '{print $4}'`
        echo -n ","${CCR} >> ${RESULTSDIR}/test_bcn_char_cr.csv
    done
done

test_ccr=$(tail -n 1 ${RESULTSDIR}/test_bcn_char_cr.csv |awk -F"," '{print ($2+$3+$4+$5+$6+$7+$8+$9+$10) / 9}')
echo ",${test_ccr}" >> ${RESULTSDIR}/test_bcn_char_cr.csv

i=0
for name in "${NAMES[@]}"; do
    NAME=${NAMES[$i]}
    echo ${NAME}
    echo -n ${NAME} >> ${RESULTSDIR}/test_bcn_char_cr.csv
    for model_id in `seq 0 2`; do
        ROOTDIR=${ROOTDIRS[$i]}_${model_id}
        python ${SCRIPTSDIR}/conv2text.py -t ${DATADIR}/test.cln.sent -l ${ROOTDIR}/test_result_greedy.bcn.sents > ${ROOTDIR}/test.bcn.comp &
    done
    wait
    for data_id in `seq 1 3`; do
        for model_id in `seq 0 2`; do
            ROOTDIR=${ROOTDIRS[$i]}_${model_id}
            CCR=`python ${SCRIPTSDIR}/eval_char_len.py -i ${DATADIR}/test.cln.strip.sent -s ${ROOTDIR}/test.bcn.comp |grep 'Compression Ratio '|awk -F" " '{print $4}'`
            echo -n ","${CCR} >> ${RESULTSDIR}/test_bcn_char_cr.csv
        done
    done
    test_ccr=$(tail -n 1 ${RESULTSDIR}/test_bcn_char_cr.csv |awk -F"," '{print ($2+$3+$4+$5+$6+$7+$8+$9+$10) / 9}')
    echo ",${test_ccr}" >> ${RESULTSDIR}/test_bcn_char_cr.csv

    let i++

done

# Long sentences

echo "Model,1-1,1-2,1-3,2-1,2-2,2-3,3-1,3-2,3-3,AVG" > ${RESULTSDIR}/test_bcn_long_char_cr.csv

# gold compression ratio

echo -n "gold" >> ${RESULTSDIR}/test_bcn_long_char_cr.csv
for data_id in `seq 1 3`; do
    for cnt in `seq 1 3`; do
        CCR=`python ${SCRIPTSDIR}/eval_char_len.py -l 20 -i ${DATADIR}/test.cln.strip.sent -s ${DATADIR}/test.cln.${data_id}.gold |grep 'Compression Ratio '|awk -F" " '{print $4}'`
        echo -n ","${CCR} >> ${RESULTSDIR}/test_bcn_long_char_cr.csv
    done
done
test_ccr=$(tail -n 1 ${RESULTSDIR}/test_bcn_long_char_cr.csv |awk -F"," '{print ($2+$3+$4+$5+$6+$7+$8+$9+$10) / 9}')
echo ",${test_ccr}" >> ${RESULTSDIR}/test_bcn_long_char_cr.csv

i=0
for name in "${NAMES[@]}"; do
    NAME=${NAMES[$i]}
    echo ${NAME}
    echo -n ${NAME} >> ${RESULTSDIR}/test_bcn_long_char_cr.csv
    for data_id in `seq 1 3`; do
        for model_id in `seq 0 2`; do
            ROOTDIR=${ROOTDIRS[$i]}_${model_id}
            CCR=`python ${SCRIPTSDIR}/eval_char_len.py -l 20 -i ${DATADIR}/test.cln.strip.sent -s ${ROOTDIR}/test.bcn.comp |grep 'Compression Ratio '|awk -F" " '{print $4}'`
            echo -n ","${CCR} >> ${RESULTSDIR}/test_bcn_long_char_cr.csv
        done
    done
    test_ccr=$(tail -n 1 ${RESULTSDIR}/test_bcn_long_char_cr.csv |awk -F"," '{print ($2+$3+$4+$5+$6+$7+$8+$9+$10) / 9}')
    echo ",${test_ccr}" >> ${RESULTSDIR}/test_bcn_long_char_cr.csv

    let i++

done

