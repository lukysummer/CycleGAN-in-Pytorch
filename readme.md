# Implementation of CycleGAN (with Skip Connection) in PyTorch

This is my own implementation of CycleGAN using PyTorch, introduced in [this paper](https://arxiv.org/pdf/1703.10593.pdf).
The main task was to carry out image-to-image translation from Horse to Zebra.

## Results

Some examples of the result:
<img src="notebook_images/skip_A2B.png">

## Datasets

Datasets necessary for CycleGAN projects can be downloaded from [this link](http://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/).

## List of Hyperparameters used:

* Batch Size = **1**
* Image Size = **128**  (128 x 128 image)
* # of filters in Discriminator's first hidden layer = **64**
* # of filters in Generator's first hidden layer = **64**
* Initial Learning Rate = **0.0002**
* # of Epochs: **55**

