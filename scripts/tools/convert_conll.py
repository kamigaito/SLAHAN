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
    parser = argparse.ArgumentParser(description='Converter')
    parser.add_argument("-i", "--in-txt", dest="in_txt", type=str, help="input file")
    parser.add_argument("-o", "--out-txt", dest="out_txt", type=str, help="output file for texts")
    parser.add_argument("-l", "--out-label", dest="out_label", type=str, help="output file for labels")
    parser.add_argument("-m", "--max-len", dest="max_len", type=int, help="maximum length limitation.", default=-1)
    parser.add_argument("-d", "--delimiter", dest="delim", type=str, help="dekimiter symbol", default="\t")
    args = parser.parse_args()
    return args

def main():
    args = options()
    with codecs.open(args.in_txt,"r",encoding="utf8") as f_in, codecs.open(args.out_txt,"w",encoding="utf8") as f_out_txt, open(args.out_label,"w",encoding="utf8") as f_out_label:
        txt_body = ""
        label_body = ""
        for line in f_in:
            line = line.strip()
            col = line.split(args.delim)
            if len(col) != 2:
                if txt_body != "" and label_body != "":
                    if args.max_len == -1 or len(txt_body.split(" ")) <= args.max_len:
                        f_out_txt.write("<s> " + txt_body + " </s>\n")
                        f_out_label.write("<s> " + label_body + " </s>\n")
                txt_body = ""
                label_body = ""
                continue
            if _clean_text(col[0]) == "":
                continue
            if txt_body != "" and label_body != "":
                txt_body += " "
                label_body += " "
            txt_body += col[0]
            if col[1] == "O":
                label_body += "1"
            else:
                label_body += "0"

if __name__ == "__main__":
    main()
