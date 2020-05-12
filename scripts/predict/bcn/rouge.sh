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

# test size

TEST_SIZE=`cat ${DATADIR}/test.cln.sent| wc -l`
echo ${TEST_SIZE}

for data_id in `seq 1 3`; do
    # Reference
    REFDIR=${XMLDIR}/ref.${data_id}
    mkdir -p ${REFDIR}/html
    for test_id in `seq 1 ${TEST_SIZE}`; do
        echo '<html>' > ${REFDIR}/html/${test_id}.html
        echo '<head>' >> ${REFDIR}/html/${test_id}.html
        echo '<title>'${test_id}'</title>' >> ${REFDIR}/html/${test_id}.html
        echo '</head>' >> ${REFDIR}/html/${test_id}.html
        echo '<body bgcolor="white">' >> ${REFDIR}/html/${test_id}.html
        #echo '<a name="'${test_id}'">['${test_id}']</a> <a href="#'${test_id}'" id='${test_id}'>' >> ${REFDIR}/html/${test_id}.html
        echo -n '<a name="1">[1]</a> <a href="#1" id=1>' >> ${REFDIR}/html/${test_id}.html
        cat ${DATADIR}/test.cln.${data_id}.gold |\
        head -n ${test_id} |\
        tail -n 1 \
        >> ${REFDIR}/html/${test_id}.html
        echo '</a>' >> ${REFDIR}/html/${test_id}.html 
        echo '</body>' >> ${REFDIR}/html/${test_id}.html 
        echo '</html>' >> ${REFDIR}/html/${test_id}.html 
    done
done

echo "Model,1-1,1-2,1-3,2-1,2-2,2-3,3-1,3-2,3-3,AVG" > ${RESULTSDIR}/test_bcn_r1.csv
echo "Model,1-1,1-2,1-3,2-1,2-2,2-3,3-1,3-2,3-3,AVG" > ${RESULTSDIR}/test_bcn_r2.csv
echo "Model,1-1,1-2,1-3,2-1,2-2,2-3,3-1,3-2,3-3,AVG" > ${RESULTSDIR}/test_bcn_rl.csv

i=0
for name in "${NAMES[@]}"; do
    NAME=${NAMES[$i]}
    echo ${NAME}
    for data_id in `seq 1 3`; do
        REFDIR=${XMLDIR}/ref.${data_id}
        for model_id in `seq 0 2`; do
            ROOTDIR=${ROOTDIRS[$i]}_${model_id}
            python ${SCRIPTSDIR}/truncate_wo_label.py -i ${DATADIR}/test.cln.strip.sent -r ${DATADIR}/test.cln.${data_id}.gold -s ${ROOTDIR}/test.bcn.conv > ${ROOTDIR}/test.limited.bcn.comp &
        done
        wait
        # Calc Rouge
        OUTDIR=${XMLDIR}/sys.${data_id}/${NAME}
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
            # HTMLl
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
                cat ${ROOTDIR}/test.limited.bcn.comp |\
                head -n ${test_id} |\
                tail -n 1 \
                >> ${OUTDIR}/${model_id}/html/${test_id}.html
                echo '</a>' >> ${OUTDIR}/${model_id}/html/${test_id}.html 
                echo '</body>' >> ${OUTDIR}/${model_id}/html/${test_id}.html 
                echo '</html>' >> ${OUTDIR}/${model_id}/html/${test_id}.html 
                echo ${NAME} ${model_id} ${test_id}'.html'
            done
            perl ./ROUGE-1.5.5/ROUGE-1.5.5.pl -e ./ROUGE-1.5.5/data -n 2 -m -d -a ${OUTDIR}/${model_id}/eval.xml > ${ROOTDIR}/rouge.bcn.${data_id}.out
        done
    done
    echo -n ${NAME} >> ${RESULTSDIR}/test_bcn_r1.csv
    echo -n ${NAME} >> ${RESULTSDIR}/test_bcn_r2.csv
    echo -n ${NAME} >> ${RESULTSDIR}/test_bcn_rl.csv
    for data_id in `seq 1 3`; do
        OUTDIR=${XMLDIR}/sys.${data_id}/${NAME}
        for model_id in `seq 0 2`; do
            ROOTDIR=${ROOTDIRS[$i]}_${model_id}
            ROUGE1=`cat ${ROOTDIR}/rouge.bcn.${data_id}.out |grep 'ROUGE-1 Eval' |awk -F" " '{print $5}' |awk -F":" '{R+=$2}END{print R/NR}'`
            ROUGE2=`cat ${ROOTDIR}/rouge.bcn.${data_id}.out |grep 'ROUGE-2 Eval' |awk -F" " '{print $5}' |awk -F":" '{R+=$2}END{print R/NR}'`
            ROUGEL=`cat ${ROOTDIR}/rouge.bcn.${data_id}.out |grep 'ROUGE-L Eval' |awk -F" " '{print $5}' |awk -F":" '{R+=$2}END{print R/NR}'`
            echo -n ","${ROUGE1} >> ${RESULTSDIR}/test_bcn_r1.csv
            echo -n ","${ROUGE2} >> ${RESULTSDIR}/test_bcn_r2.csv
            echo -n ","${ROUGEL} >> ${RESULTSDIR}/test_bcn_rl.csv
        done
    done
    test_r1=$(tail -n 1 ${RESULTSDIR}/test_bcn_r1.csv |awk -F"," '{print ($2+$3+$4+$5+$6+$7+$8+$9+$10) / 9}')
    test_r2=$(tail -n 1 ${RESULTSDIR}/test_bcn_r2.csv |awk -F"," '{print ($2+$3+$4+$5+$6+$7+$8+$9+$10) / 9}')
    test_rl=$(tail -n 1 ${RESULTSDIR}/test_bcn_rl.csv |awk -F"," '{print ($2+$3+$4+$5+$6+$7+$8+$9+$10) / 9}')
    echo ",${test_r1}" >> ${RESULTSDIR}/test_bcn_r1.csv
    echo ",${test_r2}" >> ${RESULTSDIR}/test_bcn_r2.csv
    echo ",${test_rl}" >> ${RESULTSDIR}/test_bcn_rl.csv

    let i++

done

# Long sentences

# test size
TEST_SIZE=`cat ${DATADIR}/test.long.cln.1.gold| wc -l`
echo ${TEST_SIZE}

for data_id in `seq 1 3`; do
    # Reference
    REFDIR=${XMLDIR}/ref.${data_id}
    mkdir -p ${REFDIR}/html
    for test_id in `seq 1 ${TEST_SIZE}`; do
        echo '<html>' > ${REFDIR}/html/${test_id}.html
        echo '<head>' >> ${REFDIR}/html/${test_id}.html
        echo '<title>'${test_id}'</title>' >> ${REFDIR}/html/${test_id}.html
        echo '</head>' >> ${REFDIR}/html/${test_id}.html
        echo '<body bgcolor="white">' >> ${REFDIR}/html/${test_id}.html
        #echo '<a name="'${test_id}'">['${test_id}']</a> <a href="#'${test_id}'" id='${test_id}'>' >> ${REFDIR}/html/${test_id}.html
        echo -n '<a name="1">[1]</a> <a href="#1" id=1>' >> ${REFDIR}/html/${test_id}.html
        cat ${DATADIR}/test.long.cln.${data_id}.gold |\
        head -n ${test_id} |\
        tail -n 1 \
        >> ${REFDIR}/html/${test_id}.html
        echo '</a>' >> ${REFDIR}/html/${test_id}.html 
        echo '</body>' >> ${REFDIR}/html/${test_id}.html 
        echo '</html>' >> ${REFDIR}/html/${test_id}.html 
    done
done

echo "Model,1-1,1-2,1-3,2-1,2-2,2-3,3-1,3-2,3-3,AVG" > ${RESULTSDIR}/test_bcn_r1_long.csv
echo "Model,1-1,1-2,1-3,2-1,2-2,2-3,3-1,3-2,3-3,AVG" > ${RESULTSDIR}/test_bcn_r2_long.csv
echo "Model,1-1,1-2,1-3,2-1,2-2,2-3,3-1,3-2,3-3,AVG" > ${RESULTSDIR}/test_bcn_rl_long.csv

i=0
for name in "${NAMES[@]}"; do
    NAME=${NAMES[$i]}
    echo ${NAME}
    for data_id in `seq 1 3`; do
        REFDIR=${XMLDIR}/ref.${data_id}
        for model_id in `seq 0 2`; do
            ROOTDIR=${ROOTDIRS[$i]}_${model_id}
            python ${SCRIPTSDIR}/truncate_wo_label.py -c 20 -i ${DATADIR}/test.cln.strip.sent -r ${DATADIR}/test.cln.${data_id}.gold -s ${ROOTDIR}/test.bcn.conv > ${ROOTDIR}/test.limited.bcn.long.comp &
        done
        wait
        # Calc Rouge
        OUTDIR=${XMLDIR}/sys.${data_id}/${NAME}
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
            # HTMLl
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
                cat ${ROOTDIR}/test.limited.bcn.long.comp |\
                head -n ${test_id} |\
                tail -n 1 \
                >> ${OUTDIR}/${model_id}/html/${test_id}.html
                echo '</a>' >> ${OUTDIR}/${model_id}/html/${test_id}.html 
                echo '</body>' >> ${OUTDIR}/${model_id}/html/${test_id}.html 
                echo '</html>' >> ${OUTDIR}/${model_id}/html/${test_id}.html 
                echo ${NAME} ${model_id} ${test_id}'.html'
            done
            perl ./ROUGE-1.5.5/ROUGE-1.5.5.pl -e ./ROUGE-1.5.5/data -n 2 -m -d -a ${OUTDIR}/${model_id}/eval.xml > ${ROOTDIR}/rouge.bcn.${data_id}.long.out
        done
    done
    echo -n ${NAME} >> ${RESULTSDIR}/test_bcn_r1_long.csv
    echo -n ${NAME} >> ${RESULTSDIR}/test_bcn_r2_long.csv
    echo -n ${NAME} >> ${RESULTSDIR}/test_bcn_rl_long.csv
    for data_id in `seq 1 3`; do
        OUTDIR=${XMLDIR}/sys.${data_id}/${NAME}
        for model_id in `seq 0 2`; do
            ROOTDIR=${ROOTDIRS[$i]}_${model_id}
            ROUGE1=`cat ${ROOTDIR}/rouge.bcn.${data_id}.long.out |grep 'ROUGE-1 Eval' |awk -F" " '{print $5}' |awk -F":" '{R+=$2}END{print R/NR}'`
            ROUGE2=`cat ${ROOTDIR}/rouge.bcn.${data_id}.long.out |grep 'ROUGE-2 Eval' |awk -F" " '{print $5}' |awk -F":" '{R+=$2}END{print R/NR}'`
            ROUGEL=`cat ${ROOTDIR}/rouge.bcn.${data_id}.long.out |grep 'ROUGE-L Eval' |awk -F" " '{print $5}' |awk -F":" '{R+=$2}END{print R/NR}'`
            echo -n ","${ROUGE1} >> ${RESULTSDIR}/test_bcn_r1_long.csv
            echo -n ","${ROUGE2} >> ${RESULTSDIR}/test_bcn_r2_long.csv
            echo -n ","${ROUGEL} >> ${RESULTSDIR}/test_bcn_rl_long.csv
        done
    done
    test_r1=$(tail -n 1 ${RESULTSDIR}/test_bcn_r1_long.csv |awk -F"," '{print ($2+$3+$4+$5+$6+$7+$8+$9+$10) / 9}')
    test_r2=$(tail -n 1 ${RESULTSDIR}/test_bcn_r2_long.csv |awk -F"," '{print ($2+$3+$4+$5+$6+$7+$8+$9+$10) / 9}')
    test_rl=$(tail -n 1 ${RESULTSDIR}/test_bcn_rl_long.csv |awk -F"," '{print ($2+$3+$4+$5+$6+$7+$8+$9+$10) / 9}')
    echo ",${test_r1}" >> ${RESULTSDIR}/test_bcn_r1_long.csv
    echo ",${test_r2}" >> ${RESULTSDIR}/test_bcn_r2_long.csv
    echo ",${test_rl}" >> ${RESULTSDIR}/test_bcn_rl_long.csv

    let i++

done
