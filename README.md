# Recurrent Neural Networks
This repository contains a tutorial for understanding the theory behind Recurrent Neural Networks (RNNs) and their applications. The tutorial is adapted from the [PyTorch Seq2Seq tutorial by Ben Trevett](https://github.com/bentrevett/pytorch-seq2seq).

## Tutorials

1. [Understanding Recurrent Neural Networks](https://github.com/udeepam/rnn/blob/master/1_rnn.ipynb)

In this first tutorial we cover the inner workings of the Vanilla Recurrent Neural Network and it's limitations. We then discuss the Long Short-Term Memory model as a potential solution to the vanishing gradients problem in RNNs.

2. [Language Modelling using a Recurrent Neural Network](https://github.com/udeepam/rnn/blob/master/2_language_model.ipynb)

Here we cover the application of LSTMs for the task of Language Modelling. Additionally, we introduce the basics of using [PyTorch](https://pytorch.org/) and [TorchText](https://pytorch.org/text/).

3. [Sequence to Sequence Learning with Neural Networks](https://github.com/udeepam/rnn/blob/master/3_seq2seq.ipynb)

Finally, we cover seq2seq models, and more specifically RNN encoder-decoder architectures. We apply this model to the task of Machine Translation. This is based of [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation by Cho et al.](https://arxiv.org/abs/1406.1078)