# Feed Forward Neural Network Architecture Determination

> Stop guessing the number of neurons in a layer!


## Introduction
Choosing the right architecture for your deep learning model can drastically change the results achieved. Using too few neurons can lead to the model not finding complex relationships in the data, whereas using too many neurons can lead to an overfitting effect.

With tabular data it is usually understood that not many layers are required, one or two will suffice. To help understand why this is enough look at the [Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem), which proves (in simple terms) that a neural network with one layer and a finite number of neurons can approximate any continuous function.

> However, how do you pick the number of neurons in that layer?

<br>

## Determining The Number of Neurons
The aim is to find the right number of neurons to prevent overfitting and underfitting, of course this is not the entire solution to prevent overfitting and underfitting but it can help reduce this from happening.

In machine and deep learning, the degrees of freedom in a model relate to the number of parameters which can be _learned_. Increasing the degrees of freedom in your model can give it more flexibility to fit more complex functions, but too much and this flexibility could allow the model to overfit to the data.

> This means one way to reduce overfitting is to limit the degrees of freedom in the model.

One formula I found during research of this problem is:

$$ N_{h} = \frac{N{s}}{\alpha (N_{i} + N_{o})} $$

$N_{h}$ is the number of neurons, $N_{s}$ the number of training samples $N_{i}$ represents the number of input neurons, $N_{o}$ is the number of output neurons and $\alpha$ is a hyperparameter to be selected.

The degrees of freedom in your dataset is $N_{s}(N_{i} + N_{o})$, and the aim is to limit the number of free parameters in your model to a small proportion of the degrees of freedom in your data. If your data is a good representation, this should allow the model to _generalise_ well, too many parameters and it means the model can _overfit_ to the training set.

$\alpha$ represents how many more degrees of freedom there are in your data compared to your model. With $\alpha = 2$, there will be twice as many degrees of freedom in your data than in your model. An $\alpha$ value of $2–10$ is recommended and you could loop through to find an optimal $\alpha$ value.

One method which helped me understand the formula intuitively was to let $\beta = 1 / \alpha$ (keeping $\alpha$ between $2$ and $10$) and then increasing $\beta$ represents an increasing complexity. A more nonlinear problem will need a larger beta.

> The main rule is to keep alpha ≥ 1 as this means the model degrees of freedom is never greater than the degrees of freedom in the dataset.

<br>

## Example
The example is a Jupyter Notebook and compares the advice given by _Jeff Heaton_ when selecting the number of neurons in a layer. I chose this advice as beginners are likely to follow Jeff Heaton’s advice due to his importance in the field.

This in no way takes away from Jeff Heaton’s advice but attempts to show how selecting the number of neurons using the formula provided in this blog post, a data scientist can create more complex neural networks without overfitting. _Creating more complex neural networks can lead to better results_.

Jeff Heaton’s advice:

> 1. The number of hidden neurons should be between the size of the input layer and the size of the output layer.
> 2. The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
> 3. The number of hidden neurons should be less than twice the size of the input layer.


{% gist a4d27b27689c6bae1efedc02e96c2629 %}

<br>

## References
[1]  Reference for the formula from [StackExchange](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw) 

[2] Jeff Heaton. _The Number of Hidden Layers_. __Heaton Research__. https://www.heatonresearch.com/2017/06/01/hidden-layers.html

[3] Jeremy Howard and Sylvain Gugger. fastai: A Layered API for Deep Learning. arXiv. arXiv:2002.04688 https://arxiv.org/abs/2002.04688. 2020.
