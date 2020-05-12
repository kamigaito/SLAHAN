import argparse
import numpy as np
import h5py
import codecs

# Option List
def option_parser():
    usage = "Usage (train): python {} -i /path/to/input.txt -m path/to/model -e path/to/elmo.hdf5"
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('-i', '--input_file', type=str, default="", help="Input file path")
    parser.add_argument('-o', '--output_file', type=str, default="", help="Output file path")
    parser.add_argument('-m', '--model_file', type=str, default="", help="Model file path")
    return parser.parse_args()

def main():
    args = option_parser()
    vocab = {}
    vectors = {}
    dim_size = 0
    # construct dictionaries
    with codecs.open(args.input_file, "r", encoding="utf8") as f_in:
        for line in f_in:
            line = line.strip()
            for tok in line.split(" "):
                if tok in vocab:
                    vocab[tok] += 1
                else:
                    vocab[tok] = 1
    # extract word vectors
    with codecs.open(args.model_file, "r", encoding="utf8") as f_in:
        for line in f_in:
            line = line.strip()
            cols = line.split(" ")
            dim_size = len(cols) - 1
            if cols[0] in vocab:
                vectors[cols[0]] = np.array([ float(col) for col in cols[1:]])
    # construct dictionaries
    with codecs.open(args.input_file, "r", encoding="utf8") as f_in, h5py.File(args.output_file, 'w') as f_out:
        sid = 0
        for line in f_in:
            sent_vecs = []
            line = line.strip()
            for tok in line.split(" "):
                if tok in vectors:
                    sent_vecs.append(vectors[tok])
                else:
                    sent_vecs.append(np.zeros(dim_size, dtype="float32"))
            f_out.create_dataset(str(sid), data=[sent_vecs], dtype="float32")
            sid += 1

if __name__ == "__main__":
    main()
