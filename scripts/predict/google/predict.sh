#!/bin/bash -u

DATADIR=${PWD}/dataset/conv
MODELDIR=${PWD}/models
BINDIR=${PWD}/build_compressor
SCRIPTSDIR=${PWD}/scripts/tools
RESULTSDIR=${PWD}/results

if [ ! -d ${RESULTSDIR} ]; then
    mkdir ${RESULTSDIR}
fi

# models

NAMES=("lstm")
ROOTDIRS=("${MODELDIR}/lstm")
BINS=(${BINDIR}/lstm)
NAMES=("${NAMES[@]}" "lstm-dep")
ROOTDIRS=("${ROOTDIRS[@]}" "${MODELDIR}/lstm-dep")
BINS=("${BINS[@]}" ${BINDIR}/lstm)
NAMES=("${NAMES[@]}" "attn")
ROOTDIRS=("${ROOTDIRS[@]}" "${MODELDIR}/attn")
BINS=("${BINS[@]}" ${BINDIR}/attn)
NAMES=("${NAMES[@]}" "tagger")
ROOTDIRS=("${ROOTDIRS[@]}" "${MODELDIR}/tagger")
BINS=("${BINS[@]}" ${BINDIR}/tagger)
NAMES=("${NAMES[@]}" "base")
ROOTDIRS=("${ROOTDIRS[@]}" "${MODELDIR}/base")
BINS=("${BINS[@]}" ${BINDIR}/base)
NAMES=("${NAMES[@]}" "slahan_w_syn")
ROOTDIRS=("${ROOTDIRS[@]}" "${MODELDIR}/slahan_w_syn")
BINS=("${BINS[@]}" ${BINDIR}/slahan)
NAMES=("${NAMES[@]}" "parent_w_syn")
ROOTDIRS=("${ROOTDIRS[@]}" "${MODELDIR}/parent_w_syn")
BINS=("${BINS[@]}" ${BINDIR}/slahan)
NAMES=("${NAMES[@]}" "child_w_syn")
ROOTDIRS=("${ROOTDIRS[@]}" "${MODELDIR}/child_w_syn")
BINS=("${BINS[@]}" ${BINDIR}/slahan)
NAMES=("${NAMES[@]}" "slahan_wo_syn")
ROOTDIRS=("${ROOTDIRS[@]}" "${MODELDIR}/slahan_wo_syn")
BINS=("${BINS[@]}" ${BINDIR}/slahan)
NAMES=("${NAMES[@]}" "parent_wo_syn")
ROOTDIRS=("${ROOTDIRS[@]}" "${MODELDIR}/parent_wo_syn")
BINS=("${BINS[@]}" ${BINDIR}/slahan)
NAMES=("${NAMES[@]}" "child_wo_syn")
ROOTDIRS=("${ROOTDIRS[@]}" "${MODELDIR}/child_wo_syn")
BINS=("${BINS[@]}" ${BINDIR}/slahan)

echo "Model,1,2,3,AVG" > ${RESULTSDIR}/dev_google_p.csv
echo "Model,1,2,3,AVG" > ${RESULTSDIR}/dev_google_r.csv
echo "Model,1,2,3,AVG" > ${RESULTSDIR}/dev_google_f.csv
echo "Model,1,2,3,AVG" > ${RESULTSDIR}/dev_google_cr.csv
echo "Model,1,2,3,AVG" > ${RESULTSDIR}/dev_google_em.csv

echo "Model,1,2,3,AVG" > ${RESULTSDIR}/test_google_p.csv
echo "Model,1,2,3,AVG" > ${RESULTSDIR}/test_google_r.csv
echo "Model,1,2,3,AVG" > ${RESULTSDIR}/test_google_f.csv
echo "Model,1,2,3,AVG" > ${RESULTSDIR}/test_google_cr.csv

# For all sentences
i=0
for name in "${NAMES[@]}"; do
    NAME=${NAMES[$i]}
    BIN=${BINS[$i]}
    echo ${NAME}
    echo ${BIN}
    if [ ! -d ${ROOTDIRS[$i]}_0 ]; then
        let i++
        continue
    fi
    for model_id in `seq 0 2`; do
        ROOTDIR=${ROOTDIRS[$i]}_${model_id}
        echo ${ROOTDIR}
        max_em=0
        max_id=0
        for cur_id in `seq 1 20`; do
            if [ ! -e ${ROOTDIR}/dev_${cur_id}.txt ]; then
                continue
            fi
            cur_em=$(python ${SCRIPTSDIR}/eval_overlap_macro.py -i ${DATADIR}/dev.cln.sent -r ${DATADIR}/dev.cln.label -s ${ROOTDIR}/dev_${cur_id}.txt |grep ^Exact |sed 's/ //g'|cut -d':' -f 2)
            gt=`echo ${cur_em}" > "${max_em} |bc -l`
            #echo ${cur_id}
            #echo ${cur_em}" > "${max_em}
            if [ ${gt} == 1 ]; then
                max_id=${cur_id}
                max_em=${cur_em}
            fi
        done
        # output
        cp ${ROOTDIR}/dev_${max_id}.txt ${ROOTDIR}/dev.label
        echo ${max_id} > ${ROOTDIR}/dev.info 
        python ${SCRIPTSDIR}/eval_overlap_macro.py -i ${DATADIR}/dev.cln.sent -r ${DATADIR}/dev.cln.label -s ${ROOTDIR}/dev_${max_id}.txt >> ${ROOTDIR}/dev.info
        # decode
        time ${BIN} \
        --mode predict \
        --rootdir ${ROOTDIR} \
        --modelfile <(gzip -dc ${ROOTDIR}/save_epoch_${max_id}.model.gz) \
        --srcfile ${DATADIR}/test.cln.sent \
        --trgfile ${ROOTDIR}/test_result_greedy \
        --alignfile ${DATADIR}/test.cln.dep \
        --elmo_hdf5_files ${DATADIR}/test.cln.strip.sent.glove.hdf5,${DATADIR}/test.cln.strip.sent.elmo.hdf5,${DATADIR}/test.cln.strip.sent.bert.hdf5 \
        --elmo_hdf5_dims 300,1024,768 \
        --elmo_hdf5_layers 1,3,12 \
        --guided_alignment 1 \
        --max_batch_pred 16 \
        --sort_sent_type_pred "sort_default" \
        --batch_type_pred "same_length" \
        --shuffle_batch_type_pred "default" \
        --decoder_type "greedy" \
        --beam_size 1
    done
    for model_id in `seq 0 2`; do
        ROOTDIR=${ROOTDIRS[$i]}_${model_id}
        python ${SCRIPTSDIR}/eval_overlap_macro.py -i ${DATADIR}/test.cln.sent -r ${DATADIR}/test.cln.label -s ${ROOTDIR}/test_result_greedy.sents > ${ROOTDIR}/test.info &
        python ${SCRIPTSDIR}/eval_overlap_macro.py -i ${DATADIR}/test.cln.sent -r ${DATADIR}/test.cln.label -s ${ROOTDIR}/test_result_greedy.sents -a > ${ROOTDIR}/test.all &
    done
    wait

    echo -n ${NAME} >> ${RESULTSDIR}/dev_google_p.csv
    echo -n ${NAME} >> ${RESULTSDIR}/dev_google_r.csv
    echo -n ${NAME} >> ${RESULTSDIR}/dev_google_f.csv
    echo -n ${NAME} >> ${RESULTSDIR}/dev_google_cr.csv
    echo -n ${NAME} >> ${RESULTSDIR}/dev_google_em.csv

    echo -n ${NAME} >> ${RESULTSDIR}/test_google_p.csv
    echo -n ${NAME} >> ${RESULTSDIR}/test_google_r.csv
    echo -n ${NAME} >> ${RESULTSDIR}/test_google_f.csv
    echo -n ${NAME} >> ${RESULTSDIR}/test_google_cr.csv
    
    for model_id in `seq 0 2`; do

        ROOTDIR=${ROOTDIRS[$i]}_${model_id}

        dev_p=$(cat ${ROOTDIR}/dev.info |grep '^P' |sed 's/ //g'|cut -d':' -f 2)
        dev_r=$(cat ${ROOTDIR}/dev.info |grep '^R' |sed 's/ //g'|cut -d':' -f 2)
        dev_f=$(cat ${ROOTDIR}/dev.info |grep '^F' |sed 's/ //g'|cut -d':' -f 2)
        dev_cr=$(cat ${ROOTDIR}/dev.info |grep '^C' |sed 's/ //g'|cut -d':' -f 2)
        dev_em=$(cat ${ROOTDIR}/dev.info |grep '^E' |sed 's/ //g'|cut -d':' -f 2)

        test_p=$(cat ${ROOTDIR}/test.info |grep '^P' |sed 's/ //g'|cut -d':' -f 2)
        test_r=$(cat ${ROOTDIR}/test.info |grep '^R' |sed 's/ //g'|cut -d':' -f 2)
        test_f=$(cat ${ROOTDIR}/test.info |grep '^F' |sed 's/ //g'|cut -d':' -f 2)
        test_cr=$(cat ${ROOTDIR}/test.info |grep '^C' |sed 's/ //g'|cut -d':' -f 2)

        echo -n ",${dev_p}" >> ${RESULTSDIR}/dev_google_p.csv
        echo -n ",${dev_r}" >> ${RESULTSDIR}/dev_google_r.csv
        echo -n ",${dev_f}" >> ${RESULTSDIR}/dev_google_f.csv
        echo -n ",${dev_cr}" >> ${RESULTSDIR}/dev_google_cr.csv
        echo -n ",${dev_em}" >> ${RESULTSDIR}/dev_google_em.csv

        echo -n ",${test_p}" >> ${RESULTSDIR}/test_google_p.csv
        echo -n ",${test_r}" >> ${RESULTSDIR}/test_google_r.csv
        echo -n ",${test_f}" >> ${RESULTSDIR}/test_google_f.csv
        echo -n ",${test_cr}" >> ${RESULTSDIR}/test_google_cr.csv

    done

    dev_p=$(tail -n 1 ${RESULTSDIR}/dev_google_p.csv |awk -F"," '{print ($2+$3+$4) / 3}')
    dev_r=$(tail -n 1 ${RESULTSDIR}/dev_google_r.csv |awk -F"," '{print ($2+$3+$4) / 3}')
    dev_f=$(tail -n 1 ${RESULTSDIR}/dev_google_f.csv |awk -F"," '{print ($2+$3+$4) / 3}')
    dev_cr=$(tail -n 1 ${RESULTSDIR}/dev_google_cr.csv |awk -F"," '{print ($2+$3+$4) / 3}')
    dev_em=$(tail -n 1 ${RESULTSDIR}/dev_google_em.csv |awk -F"," '{print ($2+$3+$4) / 3}')

    test_p=$(tail -n 1 ${RESULTSDIR}/test_google_p.csv |awk -F"," '{print ($2+$3+$4) / 3}')
    test_r=$(tail -n 1 ${RESULTSDIR}/test_google_r.csv |awk -F"," '{print ($2+$3+$4) / 3}')
    test_f=$(tail -n 1 ${RESULTSDIR}/test_google_f.csv |awk -F"," '{print ($2+$3+$4) / 3}')
    test_cr=$(tail -n 1 ${RESULTSDIR}/test_google_cr.csv |awk -F"," '{print ($2+$3+$4) / 3}')

    echo ",${dev_p}" >> ${RESULTSDIR}/dev_google_p.csv
    echo ",${dev_r}" >> ${RESULTSDIR}/dev_google_r.csv
    echo ",${dev_f}" >> ${RESULTSDIR}/dev_google_f.csv
    echo ",${dev_cr}" >> ${RESULTSDIR}/dev_google_cr.csv
    echo ",${dev_em}" >> ${RESULTSDIR}/dev_google_em.csv

    echo ",${test_p}" >> ${RESULTSDIR}/test_google_p.csv
    echo ",${test_r}" >> ${RESULTSDIR}/test_google_r.csv
    echo ",${test_f}" >> ${RESULTSDIR}/test_google_f.csv
    echo ",${test_cr}" >> ${RESULTSDIR}/test_google_cr.csv

    let i++

done

# For long sentences
i=0
for name in "${NAMES[@]}"; do
    NAME=${NAMES[$i]}
    BIN=${BINS[$i]}
    echo ${NAME}
    echo ${BIN}
    if [ ! -e ${ROOTDIRS[$i]}_0 ]; then
        let i++
        continue
    fi
    for model_id in `seq 0 2`; do
        ROOTDIR=${ROOTDIRS[$i]}_${model_id}
        python ${SCRIPTSDIR}/eval_overlap_macro.py -l 30 -i ${DATADIR}/test.cln.sent -r ${DATADIR}/test.cln.label -s ${ROOTDIR}/test_result_greedy.sents > ${ROOTDIR}/test.long.info &
        python ${SCRIPTSDIR}/eval_overlap_macro.py -l 30 -i ${DATADIR}/test.cln.sent -r ${DATADIR}/test.cln.label -s ${ROOTDIR}/test_result_greedy.sents -a > ${ROOTDIR}/test.long.all &
    done
    wait

    echo -n ${NAME} >> ${RESULTSDIR}/test_google_p_long.csv
    echo -n ${NAME} >> ${RESULTSDIR}/test_google_r_long.csv
    echo -n ${NAME} >> ${RESULTSDIR}/test_google_f_long.csv
    echo -n ${NAME} >> ${RESULTSDIR}/test_google_cr_long.csv
    
    for model_id in `seq 0 2`; do

        ROOTDIR=${ROOTDIRS[$i]}_${model_id}

        test_p=$(cat ${ROOTDIR}/test.long.info |grep '^P' |sed 's/ //g'|cut -d':' -f 2)
        test_r=$(cat ${ROOTDIR}/test.long.info |grep '^R' |sed 's/ //g'|cut -d':' -f 2)
        test_f=$(cat ${ROOTDIR}/test.long.info |grep '^F' |sed 's/ //g'|cut -d':' -f 2)
        test_cr=$(cat ${ROOTDIR}/test.long.info |grep '^C' |sed 's/ //g'|cut -d':' -f 2)

        echo -n ",${test_p}" >> ${RESULTSDIR}/test_google_p_long.csv
        echo -n ",${test_r}" >> ${RESULTSDIR}/test_google_r_long.csv
        echo -n ",${test_f}" >> ${RESULTSDIR}/test_google_f_long.csv
        echo -n ",${test_cr}" >> ${RESULTSDIR}/test_google_cr_long.csv

    done

    test_p=$(tail -n 1 ${RESULTSDIR}/test_google_p_long.csv |awk -F"," '{print ($2+$3+$4) / 3}')
    test_r=$(tail -n 1 ${RESULTSDIR}/test_google_r_long.csv |awk -F"," '{print ($2+$3+$4) / 3}')
    test_f=$(tail -n 1 ${RESULTSDIR}/test_google_f_long.csv |awk -F"," '{print ($2+$3+$4) / 3}')
    test_cr=$(tail -n 1 ${RESULTSDIR}/test_google_cr_long.csv |awk -F"," '{print ($2+$3+$4) / 3}')

    echo ",${test_p}" >> ${RESULTSDIR}/test_google_p_long.csv
    echo ",${test_r}" >> ${RESULTSDIR}/test_google_r_long.csv
    echo ",${test_f}" >> ${RESULTSDIR}/test_google_f_long.csv
    echo ",${test_cr}" >> ${RESULTSDIR}/test_google_cr_long.csv

    let i++

done
