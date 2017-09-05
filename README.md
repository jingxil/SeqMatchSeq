# SeqMatchSeq in Tensorflow

This is a tensorflow implementation of SeqMatchSeq model in [Learning Natural Language Inference with LSTM](https://arxiv.org/abs/1512.08849).  

![](README/arch.png)

## Environments

* Python 3.x
* TensorFlow 1.2.x

## Pre-trained Word Vectors

I use glove.6B shared by Jeffrey Pennington et al.. It can be found at [link](https://nlp.stanford.edu/projects/glove/).

## Data

The data used is Stanford Natural Language Inference (SNLI) corpus which can be downloaded at [link](https://nlp.stanford.edu/projects/snli/).

## Usage

abandoning useless word vectors

$ python customize_embedding.py --data_dir DATA_DIR --embedding_path EMBEDDING_PATH

training

$ python natural_language_inference.py --ARG=VALUE

evaluating

$ python natural_language_inference.py --forward_only=True --ARG=VALUE

visualizing

$ tensorboard --logdir=DIR

## Results

I achieved 81.7415% correct rate on dev set (\~3 epochs). 
