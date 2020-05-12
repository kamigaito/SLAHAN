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

# test size

TEST_SIZE=`cat ${DATADIR}/test.cln.sent| wc -l`
echo ${TEST_SIZE}

python ${SCRIPTSDIR}/conv2text.py \
-l ${DATADIR}/test.cln.label \
-t ${DATADIR}/test.cln.sent \
> ${DATADIR}/test.cln.gold

# Reference
REFDIR=${XMLDIR}/ref
mkdir -p ${REFDIR}/html
for test_id in `seq 1 ${TEST_SIZE}`; do
    echo '<html>' > ${REFDIR}/html/${test_id}.html
    echo '<head>' >> ${REFDIR}/html/${test_id}.html
    echo '<title>'${test_id}'</title>' >> ${REFDIR}/html/${test_id}.html
    echo '</head>' >> ${REFDIR}/html/${test_id}.html
    echo '<body bgcolor="white">' >> ${REFDIR}/html/${test_id}.html
    echo -n '<a name="1">[1]</a> <a href="#1" id=1>' >> ${REFDIR}/html/${test_id}.html
    cat ${DATADIR}/test.cln.gold |\
    head -n ${test_id} |\
    tail -n 1 \
    >> ${REFDIR}/html/${test_id}.html
    echo '</a>' >> ${REFDIR}/html/${test_id}.html 
    echo '</body>' >> ${REFDIR}/html/${test_id}.html 
    echo '</html>' >> ${REFDIR}/html/${test_id}.html 
done

echo "Model,1,2,3,AVG" > ${RESULTSDIR}/test_google_r1.csv
echo "Model,1,2,3,AVG" > ${RESULTSDIR}/test_google_r2.csv
echo "Model,1,2,3,AVG" > ${RESULTSDIR}/test_google_rl.csv

i=0
for name in "${NAMES[@]}"; do
    NAME=${NAMES[$i]}
    echo ${NAME}
    for model_id in `seq 0 2`; do
        ROOTDIR=${ROOTDIRS[$i]}_${model_id}
        python ${SCRIPTSDIR}/conv2text.py -t ${DATADIR}/test.cln.sent -l ${ROOTDIR}/test_result_greedy.sents > ${ROOTDIR}/test.comp &
    done
    wait
    for model_id in `seq 0 2`; do
        ROOTDIR=${ROOTDIRS[$i]}_${model_id}
        python ${SCRIPTSDIR}/truncate.py -i ${DATADIR}/test.cln.sent -r ${DATADIR}/test.cln.label -s ${ROOTDIR}/test_result_greedy.sents > ${ROOTDIR}/test.limited.comp &
    done
    wait
    # Calc Rouge
    OUTDIR=${XMLDIR}/sys/${NAME}
    for model_id in `seq 0 2`; do
        ROOTDIR=${ROOTDIRS[$i]}_${model_id}
        mkdir -p ${OUTDIR}/${model_id}
        # XML
        echo ${model_id}' XML'
        echo '<ROUGE-EVAL version="1.0">' > ${OUTDIR}/${model_id}/eval.xml
        for test_id in `seq 1 ${TEST_SIZE}`; do
            echo '<EVAL ID="'${test_id}'">' >> ${OUTDIR}/${model_id}/eval.xml
            echo '<PEER-ROOT>' >> ${OUTDIR}/${model_id}/eval.xml
            echo ${OUTDIR}'/'${model_id}/html >> ${OUTDIR}/${model_id}/eval.xml
            echo '</PEER-ROOT>' >> ${OUTDIR}/${model_id}/eval.xml
            echo '<MODEL-ROOT>' >> ${OUTDIR}/${model_id}/eval.xml
            echo ${REFDIR}'/html' >> ${OUTDIR}/${model_id}/eval.xml
            echo '</MODEL-ROOT>' >> ${OUTDIR}/${model_id}/eval.xml
            echo '<INPUT-FORMAT TYPE="SEE">' >> ${OUTDIR}/${model_id}/eval.xml
            echo '</INPUT-FORMAT>' >> ${OUTDIR}/${model_id}/eval.xml
            echo '<PEERS>' >> ${OUTDIR}/${model_id}/eval.xml
            #echo '<P ID="'${model_id}'">'${OUTDIR}/${model_id}/${test_id}'.html</P>' >> ${OUTDIR}/${model_id}/eval.xml
            echo '<P ID="'${model_id}'">'${test_id}'.html</P>' >> ${OUTDIR}/${model_id}/eval.xml
            echo '</PEERS>' >> ${OUTDIR}/${model_id}/eval.xml
            echo '<MODELS>' >> ${OUTDIR}/${model_id}/eval.xml
            #echo '<M ID="0">'${REFDIR}/${test_id}'.html</M>' >> ${OUTDIR}/${model_id}/eval.xml
            echo '<M ID="A">'${test_id}'.html</M>' >> ${OUTDIR}/${model_id}/eval.xml
            echo '</MODELS>' >> ${OUTDIR}/${model_id}/eval.xml
            echo '</EVAL>' >> ${OUTDIR}/${model_id}/eval.xml
        done
        echo '</ROUGE-EVAL>' >> ${OUTDIR}/${model_id}/eval.xml
        # HTML
        echo ${model_id}' HTML'
        mkdir -p ${OUTDIR}/${model_id}/html
        for test_id in `seq 1 ${TEST_SIZE}`; do
            echo '<html>' > ${OUTDIR}/${model_id}/html/${test_id}.html
            echo '<head>' >> ${OUTDIR}/${model_id}/html/${test_id}.html
            echo '<title>'${test_id}'</title>' >> ${OUTDIR}/${model_id}/html/${test_id}.html
            echo '</head>' >> ${OUTDIR}/${model_id}/html/${test_id}.html
            echo '<body bgcolor="white">' >> ${OUTDIR}/${model_id}/html/${test_id}.html
            #echo '<a name="'${test_id}'">['${test_id}']</a> <a href="#'${test_id}'" id='${test_id}'>' >> ${OUTDIR}/${model_id}/html/${test_id}.html
            echo -n '<a name="1">[1]</a> <a href="#1" id=1>' >> ${OUTDIR}/${model_id}/html/${test_id}.html
            cat ${ROOTDIR}/test.limited.comp |\
            head -n ${test_id} |\
            tail -n 1 \
            >> ${OUTDIR}/${model_id}/html/${test_id}.html
            echo '</a>' >> ${OUTDIR}/${model_id}/html/${test_id}.html 
            echo '</body>' >> ${OUTDIR}/${model_id}/html/${test_id}.html 
            echo '</html>' >> ${OUTDIR}/${model_id}/html/${test_id}.html 
            echo ${NAME} ${model_id} ${test_id}'.html'
        done
        perl ${PWD}/ROUGE-1.5.5/ROUGE-1.5.5.pl -e ./ROUGE-1.5.5/data -n 2 -m -d -a ${OUTDIR}/${model_id}/eval.xml > ${ROOTDIR}/rouge.out
    done
    echo -n ${NAME} >> ${RESULTSDIR}/test_google_r1.csv
    echo -n ${NAME} >> ${RESULTSDIR}/test_google_r2.csv
    echo -n ${NAME} >> ${RESULTSDIR}/test_google_rl.csv
    for model_id in `seq 0 2`; do
        ROOTDIR=${ROOTDIRS[$i]}_${model_id}
        ROUGE1=`cat ${ROOTDIR}/rouge.out |grep 'ROUGE-1 Eval' |awk -F" " '{print $5}' |awk -F":" '{R+=$2}END{print R/NR}'`
        ROUGE2=`cat ${ROOTDIR}/rouge.out |grep 'ROUGE-2 Eval' |awk -F" " '{print $5}' |awk -F":" '{R+=$2}END{print R/NR}'`
        ROUGEL=`cat ${ROOTDIR}/rouge.out |grep 'ROUGE-L Eval' |awk -F" " '{print $5}' |awk -F":" '{R+=$2}END{print R/NR}'`
        echo -n ","${ROUGE1} >> ${RESULTSDIR}/test_google_r1.csv
        echo -n ","${ROUGE2} >> ${RESULTSDIR}/test_google_r2.csv
        echo -n ","${ROUGEL} >> ${RESULTSDIR}/test_google_rl.csv
    done
    test_r1=$(tail -n 1 ${RESULTSDIR}/test_google_r1.csv |awk -F"," '{print ($2+$3+$4) / 3}')
    test_r2=$(tail -n 1 ${RESULTSDIR}/test_google_r2.csv |awk -F"," '{print ($2+$3+$4) / 3}')
    test_rl=$(tail -n 1 ${RESULTSDIR}/test_google_rl.csv |awk -F"," '{print ($2+$3+$4) / 3}')
    echo ",${test_r1}" >> ${RESULTSDIR}/test_google_r1.csv
    echo ",${test_r2}" >> ${RESULTSDIR}/test_google_r2.csv
    echo ",${test_rl}" >> ${RESULTSDIR}/test_google_rl.csv

    let i++

done

# For long sentences

# test size

python ${SCRIPTSDIR}/conv2text.py \
-c 30 \
-l ${DATADIR}/test.cln.label \
-t ${DATADIR}/test.cln.sent \
> ${DATADIR}/test.long.cln.gold

TEST_SIZE=`cat ${DATADIR}/test.long.cln.gold| wc -l`
echo ${TEST_SIZE}

# Reference
REFDIR=${XMLDIR}/ref
mkdir -p ${REFDIR}/html
for test_id in `seq 1 ${TEST_SIZE}`; do
    echo '<html>' > ${REFDIR}/html/${test_id}.html
    echo '<head>' >> ${REFDIR}/html/${test_id}.html
    echo '<title>'${test_id}'</title>' >> ${REFDIR}/html/${test_id}.html
    echo '</head>' >> ${REFDIR}/html/${test_id}.html
    echo '<body bgcolor="white">' >> ${REFDIR}/html/${test_id}.html
    #echo '<a name="'${test_id}'">['${test_id}']</a> <a href="#'${test_id}'" id='${test_id}'>' >> ${REFDIR}/html/${test_id}.html
    echo -n '<a name="1">[1]</a> <a href="#1" id=1>' >> ${REFDIR}/html/${test_id}.html
    cat ${DATADIR}/test.long.cln.gold |\
    head -n ${test_id} |\
    tail -n 1 \
    >> ${REFDIR}/html/${test_id}.html
    echo '</a>' >> ${REFDIR}/html/${test_id}.html 
    echo '</body>' >> ${REFDIR}/html/${test_id}.html 
    echo '</html>' >> ${REFDIR}/html/${test_id}.html 
done

mkdir -p ${RESULTSDIR}

echo "Model,1,2,3,AVG" > ${RESULTSDIR}/test_google_long_r1.csv
echo "Model,1,2,3,AVG" > ${RESULTSDIR}/test_google_long_r2.csv
echo "Model,1,2,3,AVG" > ${RESULTSDIR}/test_google_long_rl.csv

i=0
for name in "${NAMES[@]}"; do
    NAME=${NAMES[$i]}
    echo ${NAME}
    for model_id in `seq 0 2`; do
        ROOTDIR=${ROOTDIRS[$i]}_${model_id}
        python ${SCRIPTSDIR}/conv2text.py -c 30 -t ${DATADIR}/test.cln.sent -l ${ROOTDIR}/test_result_greedy.sents > ${ROOTDIR}/test.long.comp &
    done
    wait
    for model_id in `seq 0 2`; do
        ROOTDIR=${ROOTDIRS[$i]}_${model_id}
        python ${SCRIPTSDIR}/truncate.py -c 30 -i ${DATADIR}/test.cln.sent -r ${DATADIR}/test.cln.label -s ${ROOTDIR}/test_result_greedy.sents > ${ROOTDIR}/test.limited.long.comp &
    done
    wait
    # Calc Rouge
    OUTDIR=${XMLDIR}/sys/${NAME}
    for model_id in `seq 0 2`; do
        ROOTDIR=${ROOTDIRS[$i]}_${model_id}
        mkdir -p ${OUTDIR}/${model_id}
        # XML
        echo ${model_id}' XML'
        echo '<ROUGE-EVAL version="1.0">' > ${OUTDIR}/${model_id}/eval.xml
        for test_id in `seq 1 ${TEST_SIZE}`; do
            echo '<EVAL ID="'${test_id}'">' >> ${OUTDIR}/${model_id}/eval.xml
            echo '<PEER-ROOT>' >> ${OUTDIR}/${model_id}/eval.xml
            echo ${OUTDIR}'/'${model_id}/html >> ${OUTDIR}/${model_id}/eval.xml
            echo '</PEER-ROOT>' >> ${OUTDIR}/${model_id}/eval.xml
            echo '<MODEL-ROOT>' >> ${OUTDIR}/${model_id}/eval.xml
            echo ${REFDIR}'/html' >> ${OUTDIR}/${model_id}/eval.xml
            echo '</MODEL-ROOT>' >> ${OUTDIR}/${model_id}/eval.xml
            echo '<INPUT-FORMAT TYPE="SEE">' >> ${OUTDIR}/${model_id}/eval.xml
            echo '</INPUT-FORMAT>' >> ${OUTDIR}/${model_id}/eval.xml
            echo '<PEERS>' >> ${OUTDIR}/${model_id}/eval.xml
            echo '<P ID="'${model_id}'">'${test_id}'.html</P>' >> ${OUTDIR}/${model_id}/eval.xml
            echo '</PEERS>' >> ${OUTDIR}/${model_id}/eval.xml
            echo '<MODELS>' >> ${OUTDIR}/${model_id}/eval.xml
            echo '<M ID="A">'${test_id}'.html</M>' >> ${OUTDIR}/${model_id}/eval.xml
            echo '</MODELS>' >> ${OUTDIR}/${model_id}/eval.xml
            echo '</EVAL>' >> ${OUTDIR}/${model_id}/eval.xml
        done
        echo '</ROUGE-EVAL>' >> ${OUTDIR}/${model_id}/eval.xml
        # HTML
        echo ${model_id}' HTML'
        mkdir -p ${OUTDIR}/${model_id}/html
        for test_id in `seq 1 ${TEST_SIZE}`; do
            echo '<html>' > ${OUTDIR}/${model_id}/html/${test_id}.html
            echo '<head>' >> ${OUTDIR}/${model_id}/html/${test_id}.html
            echo '<title>'${test_id}'</title>' >> ${OUTDIR}/${model_id}/html/${test_id}.html
            echo '</head>' >> ${OUTDIR}/${model_id}/html/${test_id}.html
            echo '<body bgcolor="white">' >> ${OUTDIR}/${model_id}/html/${test_id}.html
            echo -n '<a name="1">[1]</a> <a href="#1" id=1>' >> ${OUTDIR}/${model_id}/html/${test_id}.html
            cat ${ROOTDIR}/test.limited.long.comp |\
            head -n ${test_id} |\
            tail -n 1 \
            >> ${OUTDIR}/${model_id}/html/${test_id}.html
            echo '</a>' >> ${OUTDIR}/${model_id}/html/${test_id}.html 
            echo '</body>' >> ${OUTDIR}/${model_id}/html/${test_id}.html 
            echo '</html>' >> ${OUTDIR}/${model_id}/html/${test_id}.html 
            echo ${NAME} ${model_id} ${test_id}'.html'
        done
        perl ${PWD}/ROUGE-1.5.5/ROUGE-1.5.5.pl -e ./ROUGE-1.5.5/data -n 2 -m -d -a ${OUTDIR}/${model_id}/eval.xml > ${ROOTDIR}/rouge.long.out
    done
    echo -n ${NAME} >> ${RESULTSDIR}/test_google_long_r1.csv
    echo -n ${NAME} >> ${RESULTSDIR}/test_google_long_r2.csv
    echo -n ${NAME} >> ${RESULTSDIR}/test_google_long_rl.csv
    for model_id in `seq 0 2`; do
        ROOTDIR=${ROOTDIRS[$i]}_${model_id}
        ROUGE1=`cat ${ROOTDIR}/rouge.long.out |grep 'ROUGE-1 Eval' |awk -F" " '{print $5}' |awk -F":" '{R+=$2}END{print R/NR}'`
        ROUGE2=`cat ${ROOTDIR}/rouge.long.out |grep 'ROUGE-2 Eval' |awk -F" " '{print $5}' |awk -F":" '{R+=$2}END{print R/NR}'`
        ROUGEL=`cat ${ROOTDIR}/rouge.long.out |grep 'ROUGE-L Eval' |awk -F" " '{print $5}' |awk -F":" '{R+=$2}END{print R/NR}'`
        echo -n ","${ROUGE1} >> ${RESULTSDIR}/test_google_long_r1.csv
        echo -n ","${ROUGE2} >> ${RESULTSDIR}/test_google_long_r2.csv
        echo -n ","${ROUGEL} >> ${RESULTSDIR}/test_google_long_rl.csv
    done
    test_r1=$(tail -n 1 ${RESULTSDIR}/test_google_long_r1.csv |awk -F"," '{print ($2+$3+$4) / 3}')
    test_r2=$(tail -n 1 ${RESULTSDIR}/test_google_long_r2.csv |awk -F"," '{print ($2+$3+$4) / 3}')
    test_rl=$(tail -n 1 ${RESULTSDIR}/test_google_long_rl.csv |awk -F"," '{print ($2+$3+$4) / 3}')
    echo ",${test_r1}" >> ${RESULTSDIR}/test_google_long_r1.csv
    echo ",${test_r2}" >> ${RESULTSDIR}/test_google_long_r2.csv
    echo ",${test_rl}" >> ${RESULTSDIR}/test_google_long_rl.csv

    let i++

done
