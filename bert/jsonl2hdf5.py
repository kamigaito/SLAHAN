# coding=utf-8
# Copyright 2019 Hidetaka Kamigaito.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert JSON formatted BERT features to HDF5 ones."""

import sys
import argparse
import json
import tokenization
import numpy as np
import h5py

# Option List
def option_parser():
    usage = "Usage: python {} -j /path/to/input.jsonl -h path/to/output.hdf5"
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('-i', '--input', dest="input", type=str, required=True, help="Input text file")
    parser.add_argument('-j', '--jsonl', dest="jsonl",type=str, required=True, help="Input jsonl file")
    parser.add_argument('-o', '--hdf5', dest="hdf5", type=str, required=True, help="Output hdf5 file")
    parser.add_argument('-v', '--vocab_file', dest="vocab_file", required=True, type=str, default="", help="")
    parser.add_argument('--do_lowercase', dest="do_lowercase", type=bool, default=True, help="")
    parser.add_argument('--ignore_separator_tokens', dest="ignore_separator_tokens", type=bool, default=False, help="If this value is True, vectors of the separator token [SEP] will be removed from the output file. This option is only supported on the HDF5 file format.")
    parser.add_argument('--ignore_class_token', dest="ignore_class_token", type=bool, default=False, help="If this value is True, vectors of the class token [CLS] will be removed from the output file. This option is only supported on the HDF5 file format.")
    parser.add_argument('--word_embeddings', dest="word_embeddings", type=bool, default=True, help="If True, sub-word-based embeddings are merged as word-based embeddings on the output file. This option is only supported on the HDF5 file format.")
    parser.add_argument('--word_composition', dest="word_composition", type=str, default="avg", help="This value decides the merging operation for creating word embeddings. ''avg'' uses averaging of sub-word embeddings, ''sum'' uses summation of sub-word embeddings, ''left-most'' uses embeddings of left-most sub-words in words as word representations, ''right-most'' uses embeddings of right-most sub-words in words as word representations. The default value  is set to ''avg''.")
    return parser.parse_args()

def main():
    args = option_parser()
    tokenizer = tokenization.FullTokenizer(
    vocab_file=args.vocab_file, do_lower_case=args.do_lowercase)
    with open(args.jsonl, mode="r") as f_json, open(args.input, mode="r") as f_txt, h5py.File(args.hdf5, 'w') as f_out:
        for line in f_json:
            txt = f_txt.readline()
            line = line.rstrip("\r\n")
            txt = txt.rstrip("\r\n")
            sent = json.loads(line)
            print("current sentence id: " + str(sent["linex_index"]))
            aligns = [0] # [CLS]
            text_list = txt.split(" ||| ")
            text_a = text_list[0]
            text_b = None
            if len(text_list) == 2:
                text_b = text_list[1]
            # For text a
            for token in text_a.split(" "):
                orig_id = aligns[-1] + 1
                for subword in tokenizer.tokenize(token):
                    aligns.append(orig_id)
            aligns.append(aligns[-1] + 1) # [SEP]
            # For text b
            if type(text_b) != type(None):
                for token in text_b.split(" "):
                    orig_id = aligns[-1] + 1
                    for subword in tokenizer.tokenize(token):
                        aligns.append(orig_id)
            # For padding
            while len(aligns) < len(sent["features"]):
                aligns.append(aligns[-1] + 1)
            # Create a dataset of the target sentence
            d3_vec = []
            tokens = []
            for layer_id in range(4):
                d2_vec = []
                d2_vec_queue = []
                tokens_queue = []
                for tok_id in range(len(sent["features"])):
                    token = sent["features"][tok_id]["token"]
                    value = sent["features"][tok_id]["layers"][layer_id]["values"]
                    if token == "[SEP]" and sent["features"][tok_id - 1]["token"] == "[SEP]":
                        break
                    if args.ignore_separator_tokens and token == "[SEP]":
                        break
                    if args.ignore_class_token and token == "[CLS]":
                        continue
                    if args.word_embeddings:
                        d2_vec_queue.append(value)
                        tokens_queue.append(token)
                        if tok_id == len(sent["features"]) - 1 or aligns[tok_id] != aligns[tok_id + 1]:
                            if layer_id == 0:
                                word = "".join(tokens_queue)
                                tokens.append(word)
                                tokens_queue = []
                            if args.word_composition == "avg":
                                d2_vec.append([
                                    round(float(x), 6) for x in np.sum(d2_vec_queue, axis=0) / len(d2_vec_queue)
                                ])
                            elif args.word_composition == "sum":
                                d2_vec.append([
                                    round(float(x), 6) for x in np.sum(d2_vec_queue, axis=0)
                                ])
                            elif args.word_composition == "left-most":
                                d2_vec.append(d2_vec_queue[0])
                            elif args.word_composition == "right-most":
                                d2_vec.append(d2_vec_queue[-1])
                            else:
                                assert(False)
                            d2_vec_queue = []
                    else:
                        if layer_id == 0:
                            tokens.append(token)
                        d2_vec.append(sent["features"][tok_id]["layers"][layer_id]["values"])
                d3_vec.append(d2_vec)
            tokens = np.array(tokens, dtype=object)
            f_out.create_dataset(str(sent["linex_index"]), data=d3_vec, dtype="float32")  
            f_out.create_dataset("txt-" + str(sent["linex_index"]), data=tokens, dtype=h5py.special_dtype(vlen=str))  

if __name__ == "__main__":
    main()
