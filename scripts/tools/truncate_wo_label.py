import argparse
import sys
import codecs

parser = argparse.ArgumentParser(description='Evaluation script for token overwrap based  macro-averaged F1 scores.')
parser.add_argument("-r", "--reference-file", dest="file_ref", type=str, help="path to reference file")
parser.add_argument("-s", "--system-output-file", dest="file_sys", type=str, help="path to system output file")
parser.add_argument("-i", "--input-text-file", dest="file_inp", type=str, help="path to input text file")
parser.add_argument("-c", "--cut-length", dest="cut_len", type=int, default=-1, help="cut sentences of which token sizes more than this value")
args = parser.parse_args()

ref_file = codecs.open(args.file_ref, "r", encoding="utf8")
sys_file = codecs.open(args.file_sys, "r", encoding="utf8")
inp_file = codecs.open(args.file_inp, "r", encoding="utf8")
std_out = codecs.open(1, "w", encoding="utf8")

for line_ref in ref_file:
    line_inp = inp_file.readline()
    line_sys = sys_file.readline()
    line_inp = line_inp.rstrip("\r\n")
    line_ref = line_ref.rstrip("\r\n")
    line_sys = line_sys.rstrip("\r\n")
    toks_inp = line_inp.split(" ")
    toks_ref = line_ref.split(" ")
    toks_sys = line_sys.split(" ")
    if len(toks_inp) <= args.cut_len:
        continue
    comp_ref = " ".join(toks_ref)
    comp_sys = " ".join(toks_sys)
    # reference length limit
    idx = len(toks_sys)
    while len(comp_ref.encode("utf-8")) < len(comp_sys.encode("utf-8")):
        idx -= 1
        comp_sys = " ".join(toks_sys[:idx])
    print(comp_sys, file=std_out)

ref_file.close()
sys_file.close()
inp_file.close()
std_out.close()
