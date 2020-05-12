# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import unicodedata
import six
import tensorflow as tf
import codecs

def _is_control(char):
  """Checks whether `chars` is a control character."""
  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat.startswith("C"):
    return True
  return False

def _is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  # \t, \n, and \r are technically contorl characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False

def _clean_text(text):
  """Performs invalid character removal and whitespace cleanup on text."""
  output = []
  for char in text:
    cp = ord(char)
    if cp == 0 or cp == 0xfffd or _is_control(char):
      continue
    if _is_whitespace(char):
      output.append(" ")
    else:
      output.append(char)
  return "".join(output)

# The below codes are written by us
import argparse
import sys
import json

def options():
    parser = argparse.ArgumentParser(description='Sentence compression dataset cleaner')
    parser.add_argument("-i", "--input-prefix", dest="input_prefix", type=str, help="input file prefix")
    parser.add_argument("-o", "--output-prefix", dest="output_prefix", type=str, help="output file prefix")
    parser.add_argument("-m", "--max-length", dest="max_length", type=int, help="maximum length limitation.", default=200)
    args = parser.parse_args()
    return args

def main():
    args = options()
    in_sent = codecs.open(args.input_prefix + ".sent", encoding="utf8", mode="r")
    in_label = codecs.open(args.input_prefix + ".label", encoding="utf8",  mode="r")
    in_dep = codecs.open(args.input_prefix + ".dep", encoding="utf8",  mode="r")
    in_rel = codecs.open(args.input_prefix + ".rel", encoding="utf8",  mode="r")
    in_pos = codecs.open(args.input_prefix + ".pos", encoding="utf8",  mode="r")
    out_sent = codecs.open(args.output_prefix + ".sent", encoding="utf8",  mode="w")
    out_label = codecs.open(args.output_prefix + ".label", encoding="utf8",  mode="w")
    out_dep = codecs.open(args.output_prefix + ".dep", encoding="utf8",  mode="w")
    out_rel = codecs.open(args.output_prefix + ".rel", encoding="utf8",  mode="w")
    out_pos = codecs.open(args.output_prefix + ".pos", encoding="utf8",  mode="w")
    while True:
        line_sent = in_sent.readline().strip()
        if not line_sent:
            break
        line_label = in_label.readline().strip()
        line_dep = in_dep.readline().strip()
        line_rel = in_rel.readline().strip()
        line_pos = in_pos.readline().strip()
        # skip long sentences
        if args.max_length > 0 and len(line_sent.split(" ")) > args.max_length:
            continue
        # tokenize
        col_sent = line_sent.split(" ")
        col_label = line_label.split(" ")
        col_dep = line_dep.split(" ")
        col_rel = line_rel.split(" ")
        col_pos = line_pos.split(" ")
        # remove malformed characters
        cln_sent = []
        cln_label = []
        cln_rel = []
        cln_pos = []
        cln_dep = []
        drop_ids = {}
        for orig_id in range(len(col_sent)):
            if _clean_text(col_sent[orig_id]) == "":
                drop_ids[orig_id] = True
                continue
            cln_sent.append(col_sent[orig_id])
            cln_label.append(col_label[orig_id])
            cln_rel.append(col_rel[orig_id])
            cln_pos.append(col_pos[orig_id])
        for orig_id in range(len(col_dep)):
            if orig_id in drop_ids:
                continue
            parent_id, child_id = col_dep[orig_id].split("-")
            dec_parent = 0
            dec_child = 0
            removed = 0
            for drop_id in drop_ids.keys():
                if int(parent_id) == drop_id:
                    removed += 1
                if int(parent_id) > drop_id:
                    dec_parent  += 1
                if int(child_id) >= drop_id:
                    dec_child += 1
            if removed > 0:
                cln_dep.append("0-" + str(int(child_id) - dec_child))
            else:
                cln_dep.append(str(int(parent_id) - dec_parent) + "-" + str(int(child_id) - dec_child))

        out_sent.write(" ".join(cln_sent) + "\n") 
        out_label.write(" ".join(cln_label) + "\n")
        out_dep.write(" ".join(cln_dep) + "\n")
        out_rel.write(" ".join(cln_rel) + "\n")
        out_pos.write(" ".join(cln_pos) + "\n")
    in_sent.close()
    in_label.close()
    in_dep.close()
    in_rel.close()
    in_pos.close()
    out_sent.close()
    out_label.close()
    out_dep.close()
    out_rel.close()
    out_pos.close()

if __name__ == "__main__":
    main()
