#!/bin/bash -eu

ROOTDIR=${1}
SCRIPTS_DIR="${ROOTDIR}/scripts/tools"
INDIR="${ROOTDIR}/dataset2/annotator"
OUTDIR="${ROOTDIR}/dataset2/conv"

if [ ! -e ${OUTDIR} ]
then
    mkdir -p ${OUTDIR}
fi

# input text
cat ${INDIR}1/* |\
\grep '<original id=' |\
sed -e 's/<[^>]*>//g' \
>  ${OUTDIR}/test.cln.strip.sent
# post process
cat ${OUTDIR}/test.cln.strip.sent |\
sed 's/^/<s> /g' |\
sed 's/$/ <\/s>/g' \
> ${OUTDIR}/test.cln.sent

# post process
for id in `seq 1 3`; do
    # Strip start and end of sentence symbols
    cat ${INDIR}${id}/* |\
    \grep '<compressed id=' |\
    sed -e 's/<[^>]*>//g' \
    >  ${OUTDIR}/test.cln.${id}.gold
    python ${SCRIPTS_DIR}/truncate_wo_label.py \
    -i ${OUTDIR}/test.cln.strip.sent \
    -r ${OUTDIR}/test.cln.${id}.gold \
    -s ${OUTDIR}/test.cln.${id}.gold \
    -c 20 \
    > ${OUTDIR}/test.long.cln.${id}.gold
done
