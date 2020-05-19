# Compressed sentences for each model

This directory ``share/`` contains files that were used in our evaluation.

## File List

### Output results from the google sentence compression dataset:
- `./google/input.txt`: Input file
- `./google/gold.comp`: Gold compression
- `./google/tagger_0.comp`, `./google/tagger_1.comp`, `./google/tagger_2.comp`: Ouput of `Tagger` in our paper.
- `./google/lstm_0.comp`, `./google/lstm_1.comp`, `./google/lstm_2.comp`: Output of `LSTM` in our paper.
- `./google/lstm_dep_0.comp`, `./google/lstm_dep_1.comp`, `./google/lstm_dep_2.comp`: Output of `LSTM-Dep` in our paper.
- `./google/attn_0.comp`, `./google/attn_1.comp`, `./google/attn_2.comp`: Output of `Attn` in our paper.
- `./google/base_0.comp`, `./google/base_1.comp`, `./google/base_2.comp`: Output of `Base` in our paper.
- `./google/parent_w_syn_0.comp`, `./google/parent_w_syn_1.comp`, `./google/parent_w_syn_2.comp`: Output of `Parent w/ syn` in our paper.
- `./google/parent_wo_syn_0.comp`, `./google/parent_wo_syn_1.comp`, `./google/parent_wo_syn_2.comp`: Output of `Parent w/o syn` in our paper.
- `./google/child_w_syn_0.comp`, `./google/child_w_syn_1.comp`, `./google/child_w_syn_2.comp`: Output of `Child w/ syn` in our paper.
- `./google/child_wo_syn_0.comp`, `./google/child_wo_syn_1.comp`, `./google/child_wo_syn_2.comp`: Output of `Child w/o syn` in our paper.
- `./google/slahan_w_syn_0.comp`, `./google/slahan_w_syn_1.comp`, `./google/slahan_w_syn_2.comp`: Output of `Both w/ syn` in our paper.
- `./google/slahan_wo_syn_0.comp`, `./google/slahan_wo_syn_1.comp`, `./google/slahan_wo_syn_2.comp`: Output of `Both w/ syn` in our paper.

The indexes 0, 1, and 2 for each model denote the model which generated the result.

### Output results from the broadcast news commentary corpus:
- `./bcn/input.txt`: Input file
- `./bcn/gold.1.comp`, `./bcn/gold.2.comp`, `./bcn/gold.3.comp`: Gold compressions for annotator 1, 2, and 3.
- `./bcn/tagger_0.comp`, `./bcn/tagger_1.comp`, `./bcn/tagger_2.comp`: Ouput of `Tagger` in our paper.
- `./bcn/lstm_0.comp`, `./bcn/lstm_1.comp`, `./bcn/lstm_2.comp`: Output of `LSTM` in our paper.
- `./bcn/lstm_dep_0.comp`, `./bcn/lstm_dep_1.comp`, `./bcn/lstm_dep_2.comp`: Output of `LSTM-Dep` in our paper.
- `./bcn/attn_0.comp`, `./bcn/attn_1.comp`, `./bcn/attn_2.comp`: Output of `Attn` in our paper.
- `./bcn/base_0.comp`, `./bcn/base_1.comp`, `./bcn/base_2.comp`: Output of `Base` in our paper.
- `./bcn/parent_w_syn_0.comp`, `./bcn/parent_w_syn_1.comp`, `./bcn/parent_w_syn_2.comp`: Output of `Parent w/ syn` in our paper.
- `./bcn/parent_wo_syn_0.comp`, `./bcn/parent_wo_syn_1.comp`, `./bcn/parent_wo_syn_2.comp`: Output of `Parent w/o syn` in our paper.
- `./bcn/child_w_syn_0.comp`, `./bcn/child_w_syn_1.comp`, `./bcn/child_w_syn_2.comp`: Output of `Child w/ syn` in our paper.
- `./bcn/child_wo_syn_0.comp`, `./bcn/child_wo_syn_1.comp`, `./bcn/child_wo_syn_2.comp`: Output of `Child w/o syn` in our paper.
- `./bcn/slahan_w_syn_0.comp`, `./bcn/slahan_w_syn_1.comp`, `./bcn/slahan_w_syn_2.comp`: Output of `Both w/ syn` in our paper.
- `./bcn/slahan_wo_syn_0.comp`, `./bcn/slahan_wo_syn_1.comp`, `./bcn/slahan_wo_syn_2.comp`: Output of `Both w/ syn` in our paper.

The indexes 0, 1, and 2 for each model denote the model which generated the result.

## Details

### Preprocess

Following the original tokenization of the Google sentence compression dataset.
All words are not lowercased.
Some invisual spaces are removed following the subword tokenizer of BERT.

### Model Selection for the Human Evaluation

We select a model with the best F-1 score on the development dataset for the evaluation.

### Filtering for the Human Evaluation

To remove ambiguous sentences, we filtered out following pairs:
- Sentence and its compression are the same.
- Including double quotations in the sentence.
- The sentence includes a word its all characters are capitalized.
- Depend on other sentences.

After that, we further filtered out sentences whose compressions are the same for all the models and selected the first 100 sentences from the test set of the Google dataset.
