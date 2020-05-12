import argparse
import sys
import codecs

parser = argparse.ArgumentParser(description='Evaluation script for token overwrap based  macro-averaged F1 scores.')
parser.add_argument("-r", "--reference-file", dest="file_ref", type=str, help="path to reference file")
parser.add_argument("-s", "--system-output-file", dest="file_sys", type=str, help="path to system output file")
parser.add_argument("-i", "--input-text-file", dest="file_inp", type=str, help="path to input text file")
parser.add_argument("-l", "--length-limit", dest="len_limit", type=int, default=-1, help="Only evaluate sentences longer than this value. (default: -1)")
parser.add_argument("-a", "--all-score", dest="all_score", action='store_true', default=False, help="Print all scores. (default: False)")
args = parser.parse_args()

ref_file = codecs.open(args.file_ref, "r", encoding="utf8")
sys_file = codecs.open(args.file_sys, "r", encoding="utf8")
inp_file = codecs.open(args.file_inp, "r", encoding="utf8")

exact_match = 0
doc_len = 0
precision = 0
recall = 0
f = 0
comp_rate = 0

if args.all_score:
    print("Precision\tRecall\tF1\tCR")

for line_ref in ref_file:
    prec_cor = 0
    prec_all = 0
    recall_cor = 0
    recall_all = 0
    tok_all = 0
    tok_kept = 0
    line_inp = inp_file.readline()
    line_sys = sys_file.readline()
    line_inp = line_inp.rstrip("\r\n")
    line_ref = line_ref.rstrip("\r\n")
    line_sys = line_sys.rstrip("\r\n")
    toks_inp = line_inp.split(" ")
    toks_ref = line_ref.split(" ")
    toks_sys = line_sys.split(" ")
    assert len(toks_ref) == len(toks_sys)
    if len(toks_ref) < args.len_limit:
        continue
    doc_len += 1
    if toks_ref == toks_sys:
        exact_match += 1
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
    prec_local = 0.0
    recall_local = 0.0
    f_local = 0.0
    cr_local = 0.0
    if prec_all > 0:
        prec_local = prec_cor / float(prec_all)
    recall_local = recall_cor / float(recall_all)
    if prec_all > 0 and (prec_cor != 0 or recall_cor != 0):
        f_local = 2 * (prec_cor / float(prec_all) * recall_cor / float(recall_all)) / (prec_cor / float(prec_all) + recall_cor / float(recall_all))
    cr_local = tok_kept / float(tok_all)
    # Print 
    if args.all_score:
        print(str(prec_local) + "\t" + str(recall_local) + "\t" + str(f_local) + "\t" + str(cr_local))
    # Update 
    if not args.all_score:
        precision += prec_local
        recall += recall_local
        f += f_local
        comp_rate += cr_local

ref_file.close()
sys_file.close()
inp_file.close()

if not args.all_score:
    precision /= float(doc_len)
    recall /= float(doc_len)
    f /= float(doc_len)
    comp_rate /= float(doc_len)
    exact_match /= float(doc_len)
    print("Line              : " + str(doc_len))
    print("F1                : " + str(f))
    print("Precision         : " + str(precision))
    print("Recall            : " + str(recall))
    print("Compression Ratio : " + str(comp_rate))
    print("Exact Match       : " + str(exact_match))
