import argparse
import sys
import codecs

parser = argparse.ArgumentParser(description='Evaluation script for token overwrap based micro-averaged F1 scores.')
parser.add_argument("-r", "--reference-file", dest="file_ref", type=str, help="path to reference file")
parser.add_argument("-s", "--system-output-file", dest="file_sys", type=str, help="path to system output file")
parser.add_argument("-i", "--input-text-file", dest="file_inp", type=str, help="path to input text file")
args = parser.parse_args()

ref_file = codecs.open(args.file_ref, "r", encoding="utf8")
sys_file = codecs.open(args.file_sys, "r", encoding="utf8")
inp_file = codecs.open(args.file_inp, "r", encoding="utf8")

prec_cor = 0
prec_all = 0
recall_cor = 0
recall_all = 0
tok_all = 0
tok_kept = 0
sent_cor = 0
doc_len = 0
for line_ref in ref_file:
    line_inp = inp_file.readline()
    line_sys = sys_file.readline()
    line_inp = line_inp.rstrip("\r\n")
    line_ref = line_ref.rstrip("\r\n")
    line_sys = line_sys.rstrip("\r\n")
    toks_inp = line_inp.split(" ")
    toks_ref = line_ref.split(" ")
    toks_sys = line_sys.split(" ")
    assert len(toks_ref) == len(toks_sys)
    doc_len += 1
    if toks_ref == toks_sys:
        sent_cor += 1
    dict_ref = {}
    dict_sys = {}
    for i in range(0, len(toks_ref)):
        if toks_ref[i] == "1" or toks_ref[i] == "0":
            tok_all += 1
        if toks_ref[i] == "1":
            if toks_inp[i] in dict_ref:
                dict_ref[toks_inp[i]] += 1
            else:
                dict_ref[toks_inp[i]] = 1
        if toks_sys[i] == "1":
            tok_kept += 1
            if toks_inp[i] in dict_sys:
                dict_sys[toks_inp[i]] += 1
            else:
                dict_sys[toks_inp[i]] = 1
    for tok in dict_sys.keys():
        prec_all += dict_sys[tok]
        if tok in dict_ref:
            prec_cor += dict_sys[tok]
    for tok in dict_ref.keys():
        recall_all += dict_ref[tok]
        if tok in dict_sys:
            recall_cor += dict_ref[tok]

precision = prec_cor / float(prec_all)
recall = recall_cor / float(recall_all)
f = 2 * precision * recall / (precision + recall)
comp_rate = tok_kept / float(tok_all)
exact_match = sent_cor / float(doc_len)

print("Line              : " + str(doc_len))
print("F1                : " + str(f))
print("Precision         : " + str(precision))
print("Recall            : " + str(recall))
print("Compression Ratio : " + str(comp_rate))
print("Exact Match       : " + str(exact_match))
