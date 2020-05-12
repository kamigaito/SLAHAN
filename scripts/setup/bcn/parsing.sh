#!/bin/bash -eu

ROOT_DIR=${1}
IN_DIR="${ROOT_DIR}/dataset2/conv"
OUT_DIR="${ROOT_DIR}/dataset2/conv"

cd ${ROOT_DIR}/stanford-parser-full-2018-10-17
java \
-mx1500m \
-cp "./*:" \
edu.stanford.nlp.parser.lexparser.LexicalizedParser \
-outputFormat "conll2007" \
-outputFormatOptions includePunctuationDependencies \
-tokenized \
-sentences newline \
edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz \
${IN_DIR}/test.cln.strip.sent \
> ${OUT_DIR}/test.cln.strip.dep.conll

cd ${ROOT_DIR}
