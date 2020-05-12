import argparse
import codecs

def options():
    parser = argparse.ArgumentParser(description='align checker')
    parser.add_argument("-i", "--input-sent", dest="inp_sent", type=str, help="path to a sentence file")
    parser.add_argument("-d", "--input-dep", dest="inp_dep", type=str, help="path to a dependency file")
    args = parser.parse_args()
    return args

def main():
    args = options()
    with codecs.open(args.inp_sent,"r",encoding="utf8") as f_sent, codecs.open(args.inp_dep,"r",encoding="utf8") as f_dep, codecs.open(1,"w",encoding="utf8") as std_out:
        sid = 1
        for sent in f_sent:
            dep = f_dep.readline().strip()
            col_sent = sent.split(" ")
            col_dep = dep.split(" ")
            if  len(col_sent) != len(col_dep):
                print(dep, file=std_out)
                print(str(len(col_sent)) + ", " + str(len(col_dep)), file=std_out)
                print(str(sid), file=std_out)
            sid += 1

if __name__ == "__main__":
    main()
