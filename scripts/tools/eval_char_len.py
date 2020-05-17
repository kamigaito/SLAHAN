import argparse
import sys
import codecs

parser = argparse.ArgumentParser(description='Evaluation script for compression rate in characters.')
parser.add_argument("-i", "--input-text-file", dest="file_inp", type=str, help="path to an input text file")
parser.add_argument("-s", "--system-output-file", dest="file_sys", type=str, help="path to a system output file")
parser.add_argument("-l", "--length-limit", dest="len_limit", type=int, default=-1, help="Only evaluate sentences longer than this value. (default: -1)")
args = parser.parse_args()

sys_file = codecs.open(args.file_sys, "r", encoding="utf8")
inp_file = codecs.open(args.file_inp, "r", encoding="utf8")

doc_len = 0
comp_rate = 0

for line_inp in inp_file:
    line_sys = sys_file.readline()
    assert(line_sys)
    line_sys = line_sys.rstrip("\r\n")
    line_inp = line_inp.rstrip("\r\n")
    toks_inp = line_inp.split(" ")
    if len(toks_inp) < args.len_limit:
        continue
    doc_len += 1
    cr_local = len(line_sys) / len(line_inp)
    comp_rate += cr_local

comp_rate /= doc_len
print("Compression Ratio : " + str(comp_rate))
