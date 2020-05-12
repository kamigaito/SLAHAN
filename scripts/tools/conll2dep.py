# The below codes are written by us
import argparse
import sys
import codecs

def options():
    parser = argparse.ArgumentParser(description='Converter')
    parser.add_argument("-i", "--in-txt", dest="in_txt", type=str, help="input file")
    parser.add_argument("-o", "--out-txt", dest="out_txt", type=str, help="output file for texts")
    args = parser.parse_args()
    return args

def main():
    args = options()
    in_sent = open(args.in_txt, "r")
    with codecs.open(args.in_txt,"r",encoding="utf8") as f_in, codecs.open(args.out_txt,"w",encoding="utf8") as f_out_txt:
        txt_body = ""
        last_id = "0"
        for line in f_in:
            line = line.strip()
            col = line.split("\t")
            if len(col) != 10:
                if txt_body != "":
                    f_out_txt.write("0-0 " + txt_body + " 0-" + str(int(last_id) + 1) + "\n")
                txt_body = ""
                continue
            if txt_body != "":
                txt_body += " "
            txt_body += col[6] + "-" + col[0]
            last_id = col[0]

if __name__ == "__main__":
    main()
