import argparse
import sys
import codecs

parser = argparse.ArgumentParser(description='Google sentemce compression dataset converter')
parser.add_argument("-l", "--label-file", dest="file_label", type=str, help="path to label file")
parser.add_argument("-t", "--text-file", dest="file_txt", type=str, help="path to text file")
parser.add_argument("-c", "--cut-length", dest="cut_len", type=int, default=-1, help="cut sentences of which token sizes more than this value")
args = parser.parse_args()

label_file = codecs.open(args.file_label, "r", encoding="utf8")
txt_file = codecs.open(args.file_txt, "r", encoding="utf8")
std_out = codecs.open(1, 'w', encoding="utf-8")

prec_cor = 0
prec_all = 0
recall_cor = 0
recall_all = 0
tok_all = 0
tok_keep = 0

for line_label in label_file:
    line_txt = txt_file.readline()
    line_label = line_label.rstrip("\r\n")
    line_txt = line_txt.rstrip("\r\n")
    toks_label = line_label.split(" ")
    toks_txt = line_txt.split(" ")
    assert len(toks_label) == len(toks_txt)
    if len(toks_txt) < args.cut_len:
        continue
    comp_txt = ""
    tok_id = 0
    for i in range(0, len(toks_label)):
        if toks_label[i] == "1":
            if comp_txt != "":
                comp_txt += " "
            if tok_id == 0:
                comp_txt += toks_txt[i][0].upper() + toks_txt[i][1:]
            else:
                comp_txt += toks_txt[i]
            tok_id += 1
    print(comp_txt, file=std_out)
