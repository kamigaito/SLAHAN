SLAHAN: Syntactically Look-A-Head Attention Network for Sentence Compression
=============================================================================

`SLAHAN` is an implementation of Kamigaito et al., 2020, "[Syntactically Look-A-Head Attention Network for Sentence Compression](https://arxiv.org/abs/2002.01145)", In Proc. of AAAI2020.

## Citation

`````
@article{Kamigaito2020SyntacticallyLA,
  title={Syntactically Look-Ahead Attention Network for Sentence Compression},
  author={Hidetaka Kamigaito and Manabu Okumura},
  journal={ArXiv},
  year={2020},
  volume={abs/2002.01145}}
`````

## Prerequisites
- GCC 7.5.0 (supporting the C++11 language standard)
- CMake 3.14.1
- Boost Libraries version 1.63
- CUDA 10.1/10.0
- [HDF5 Library 1.10.4](https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.4/src/)
- Python 3.69
- [TensorFlow 1.15.0](https://github.com/tensorflow/tensorflow)
- [AllenNLP 0.9.0](https://github.com/allenai/allennlp)
- h5py 2.8.0
- Perl 5.26.2
- Java
- Bash

## Build Instruction
Set the environment value `BOOST_ROOT`:
`````
export BOOST_ROOT="/path/to/boost"
`````
and run:
`````
./scripts/setup.sh
`````

## Prediction & Evaluation
You can predict the test sets of Google dataset and BNC Corpus by the following commands:
`````
### Prediction for the Google dataset ###
./scripts/predict/google/predict.sh
### Prediction for the BNC Corpus ###
./scripts/predict/bcn/predict.sh
`````
Precisions, recalls, f-measures, and compression ratios for each model are stored in the file ``results/{dev or test}_{dataset size}_{evaluation metric}.csv``, respectively.
You can also calculate rouge scores by using the ``ROUGE-1.5.5`` script.
After the setup of ``ROUGE-1.5.5`` and the execution of `predict.sh`, you can ran the following command for obtaining rouge scores for each model:
`````
### Calculate rouge for the Google dataset ###
./scripts/predict/google/rouge.sh
### Calculate rouge for the BNC Corpus ###
./scripts/predict/bcn/rouge.sh
`````
The reference compression is located on ``dataset/``, and the system compression is located on ``models/{dataset size}/{model name}_{process id}/comp.txt``.

## Compressed Sentences

The compressed sentences used for the evaluation in our paper are included in the directory ``share``.

## Retrain the Models

Before retraining the models, you should extract features of training data set.
`````
./scripts/train/google/extract_features.sh
`````
Note that this step includes feature extractions of BERT, ELMo and Glove.
It takes almost 1 day and 300GB of disk space.
After that, you can retrain each model by the below command.
`````
./scripts/train/{model name}.sh {process_id}
`````

All trained models are saved in the directory ``models/{dataset name}/{model name}_{process id}/save_{epoch}``.
To reproduce our results, run the following commands:

`````
### Tagger ###
./scripts/train/tagger.sh 0
./scripts/train/tagger.sh 1
./scripts/train/tagger.sh 2
`````
`````
### LSTM ###
./scripts/train/lstm.sh 0
./scripts/train/lstm.sh 1
./scripts/train/lstm.sh 2
`````
`````
### LSTM-Dep ###
./scripts/train/lstm-dep.sh 0
./scripts/train/lstm-dep.sh 1
./scripts/train/lstm-dep.sh 2
`````
`````
### Attn ###
./scripts/train/attn.sh 0
./scripts/train/attn.sh 1
./scripts/train/attn.sh 2
`````
`````
### Base ###
./scripts/train/base.sh 0
./scripts/train/base.sh 1
./scripts/train/base.sh 2
`````
`````
### Parent w syn ###
./scripts/train/parent_w_syn.sh 0
./scripts/train/parent_w_syn.sh 1
./scripts/train/parent_w_syn.sh 2
`````
`````
### Parent w/o syn ###
./scripts/train/parent_wo_syn.sh 0
./scripts/train/parent_wo_syn.sh 1
./scripts/train/parent_wo_syn.sh 2
`````
`````
### Child w syn ###
./scripts/train/child_w_syn.sh 0
./scripts/train/child_w_syn.sh 1
./scripts/train/child_w_syn.sh 2
`````
`````
### Child w/o syn ###
./scripts/train/child_wo_syn.sh 0
./scripts/train/child_wo_syn.sh 1
./scripts/train/child_wo_syn.sh 2
`````
`````
### SLAHAN w syn ###
./scripts/train/slahan_w_syn.sh 0
./scripts/train/slahan_w_syn.sh 1
./scripts/train/slahan_w_syn.sh 2
`````
`````
### SLAHAN w/o syn ###
./scripts/train/google/slahan_wo_syn.sh 0
./scripts/train/google/slahan_wo_syn.sh 1
./scripts/train/google/slahan_wo_syn.sh 2
`````

After these processes, you can run the following prediction and evaluation scripts:
`````
./scripts/predict/google/predict.sh
./scripts/predict/google/rouge.sh
./scripts/predict/bcn/predict.sh
./scripts/predict/bcn/rouge.sh
`````
Finally, you can obtain results of the models in the directory `./results`.

## LICENSE
MIT License
