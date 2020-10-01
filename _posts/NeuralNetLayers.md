---
layout: post
title: Neural Network Architecture Determination
mathjax: true
description: Stop guessing the number of neurons in a layer!
---
>

## Introduction
Choosing the right architecture for your deep learning model can drastically change the results achieved. Using too few neurons can lead to the model not finding complex relationships in the data, whereas using too many neurons can lead to an overfitting effect.

With tabular data it is usually understood that not many layers are required, one or two will suffice. To help understand why this is enough look at the [Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem), which proves (in simple terms) that a neural network with one layer and a finite number of neurons can approximate any continuous function.

> However, how do you pick the number of neurons in that layer?

## Determining The Number of Neurons
The aim is to find the right number of neurons to prevent overfitting and underfitting, of course this is not the entire solution to prevent overfitting and underfitting but it can help reduce this from happening.

In machine and deep learning, the degrees of freedom in a model relate to the number of parameters which can be _learned_. Increasing the degrees of freedom in your model can give it more flexibility to fit more complex functions, but too much and this flexibility could allow the model to overfit to the data.

> This means one way to reduce overfitting is to limit the degrees of freedom in the model.

One formula I found during research of this problem is:

$$ N_{h} = \frac{N{s}}{\alpha (N_{i} + N_{o})} $$

$N_{h}$ is the number of neurons, $N_{s}$ the number of training samples $N_{i}$ represents the number of input neurons, $N_{o}$ is the number of output neurons and $\alpha$ is a hyperparameter to be selected.

The degrees of freedom in your dataset is $N_{s}(N_{i} + N_{o})$, and the aim is to limit the number of free parameters in your model to a small portion of the degrees of freedom in your data. If your data is a good representation, this should allow the model to generalise well, too many parameters and it means the model can overfit to the training set.

> The model is unlikely to overfit as the degrees of freedom are limited to a number proportional to the degrees of freedom of the dataset.

An $\alpha$ value of 2-10 is recommended, and it is best to loop through to find an optimal. An easy way to understand this is to let $\beta = \frac{1}{\alpha}$ and increasing $\beta$ represents an increasing complexity.

A more nonlinear problem will need a larger $\beta$. The main part is to keep $\alpha \geq 1$ as this means the number of neurons are never greater than the degrees of freedom in the dataset.

## Example

## References
https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
