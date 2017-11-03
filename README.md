# Fraternal Dropout

### Example of using fraternal dropout in a case of LSTM Language Model for PTB dataset

This repository contains the code originally forked from the [AWD-LSTM Language Model](https://github.com/salesforce/awd-lstm-lm) that is simplified and modified to present the performance of [Fraternal Dropout](https://arxiv.org/abs/1711.00066) (FD).

The architecture used here is a single layer LSTM with [DropConnect (Wan et al. 2013)](https://cs.nyu.edu/~wanli/dropc/dropc.pdf) applied on the RNN hidden to hidden matrix. The same that is used in the ablation studies in the [Fraternal Dropout](https://arxiv.org/abs/1711.00066) paper.
Included below are hyper-parameters to get equivalent results to those in the original paper for a single layer LSTM.

## Software Requirements

Python 3 and PyTorch 0.2.

## How to run the code

The easiest way to train the FD model (baseline model enchanced by fraternal dropout with κ=0.15) achiving perplexities of approximately `67.5` / `64.9` (validation / testing) is to run

+ `python main.py --model FD`

For the comparison you can try

+ `python main.py --model ELD`
or
+ `python main.py --model PM`

to train expectation-linear dropout model (κ=0.25) or Π-model (κ=0.15), respectively.

If you want to override default κ value just use `--kappa`, for example

+ `python main.py --model FD --kappa 0.1`

## Additional options

There are a few hyper-parameters you may try, run

+ `python main.py --help`

to get the full list of all of them.

For instance

+ `python main.py --model FD --same_mask_w`

use the same dropout mask for the RNN hidden to hidden matrix in both networks. That gives a little better results i.e. `66.9` / `64.6` (validation / testing).

## Using fraternal dropout in other pytorch models

With this example, it should be easy to apply fraternal dropout in any pytorch model that uses dropout. However, this example incorporates additional options (like using the same dropout mask for a part of neural network or applying expectation-linear dropout model instead of fraternal dropout) and simple example is provided below.

If you are interested in applying fraternal dropout without additional options (which are not important to achieve better results, they are implemented here just to have a comparison) just a simple modification of your code should be enough. 

You will have to find and modify the lines of code that calculate the output and loss. In the simplest, typical case you should find something like that

+ `output = model(data)`
+ `loss = criterion(output, targets)`

and replace with

+ `output = model(data)`
+ `kappa_output = model(data)`
+ `loss = 1/2*(criterion(output, targets) + criterion(kappa_output, targets))`
+ `loss = loss + kappa * (output - kappa_output).pow(2).mean()`

Since by default a new dropout mask is drawn for each forward pass, `output` and `kappa_output` are calculated for different masks and hence are not the same. You should average target loss for both of them (`loss = 1/2*(criterion(output, targets) + criterion(kappa_output, targets))`) and add regularization that makes the variance for different masks smaller (`loss = loss + kappa * (output - kappa_output).pow(2).mean()`). The variable `kappa` is a κ hyper-parameter.