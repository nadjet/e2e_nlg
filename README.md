# e2e seq2seq NLG

This is the code for doing e2e seq2seq NLG using [Fastai](https://github.com/fastai/course-nlp).

It is targeted at the [e2e NLG competition](http://www.macs.hw.ac.uk/InteractionLab/E2E/).

Medium article explaining this work is available [here](https://medium.com/@nadjetba/seq2seq-nlg-the-good-the-bad-and-the-ugly-8de0a05d9da1).

## Requirements

- python==3.7.4
- fastai==1.0.59
- fasttext==0.9.1
- torch==1.3.1

## External resources

- [e2e NLG dataset](https://github.com/tuetschek/e2e-dataset)
- [e2e NLG metrics script](https://github.com/tuetschek/e2e-metrics).
- [Fasttext English word vectors](https://fasttext.cc/docs/en/crawl-vectors.html).
- [Classifier model](https://www.kaggle.com/nadjetba/text-to-meaning-with-multi-label-classification/output) for reranking (`vocab.pkl` and `classifier_model.pth`)

## Code content

- notebooks for training and testing the models.
- Fastai code implementing the seq2seq model, and taken from [here](https://github.com/fastai/course-nlp).
- code for preprocessing texts and meaning representations before feeding it to seq2seq: cleaning, delexicalization.
- code for postprocessing texts and meaning representations before feeding them to evaluation scripts: relexicalization, displaying outputs in same order as input and references files.
- code for data augmentation, inferencing using MR classifier, and reranking.