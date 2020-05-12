import argparse
import codecs

parser = argparse.ArgumentParser(description='Conveter of the Google sentence compression dataset')
parser.add_argument("-s", "--sent-file", dest="file_sent", type=str, help="path to the sentence file")
parser.add_argument("-p", "--pos-file", dest="file_pos", type=str, help="path to the pos file")
parser.add_argument("-r", "--rel-file", dest="file_rel", type=str, help="path to the relation file")
parser.add_argument("-o", "--out-file", dest="file_out", type=str, help="path to the output file")
opts = parser.parse_args()

f_sent = codecs.open(opts.file_sent,"r",encoding="utf8")
f_pos = codecs.open(opts.file_pos,"r",encoding="utf8")
f_rel = codecs.open(opts.file_rel,"r",encoding="utf8")
f_out = codecs.open(opts.file_out,"w",encoding="utf8")

with f_sent, f_pos, f_rel, f_out:
    for line_sent in f_sent:
        line_sent = line_sent.rstrip()
        line_pos = f_pos.readline().rstrip()
        line_rel = f_rel.readline().rstrip()
        col_sent = line_sent.split(" ")
        col_pos = line_pos.split(" ")
        col_rel = line_rel.split(" ")
        if len(col_sent) != len(col_pos) or len(col_sent) != len(col_rel):
            print("POS, Rel and Tokens are not correctly aligned.")
            assert(len(col_sent) == len(col_pos))
            assert(len(col_sent) == len(col_rel))
        body = "";
        for i in range(len(col_sent)):
            if body != "":
                body += " "
            body += col_sent[i] + "-|-" + col_pos[i] + "-|-" + col_rel[i]
        f_out.write(body + "\n")
