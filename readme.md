# Reccurrent Attention Model

## Introduction
This repo is an implementation of Reccurrent Attention Model (RAM) from [Recurrent Models of Visual Attention](http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf). <!-- Here is a Chinese blog []() with a short introduction about the attention mechanism and the RAM. -->

I tested the model on $28 \times 28$ MNIST dataset and got the following results:

nums\_glimpses | error rate | error rate in the paper
---|---|---
2 | 2.34% | 3.79%
7 | 1.93% | 1.07%

## Requirements

- Python 3.6+
- PyTorch 0.4

## Usage
The code has been tested in a CPU-only environment.

See detail in `train.py` for hyperparameters setting and run the following command with arguments:

```
python train.py --epochs 30 ...
```
