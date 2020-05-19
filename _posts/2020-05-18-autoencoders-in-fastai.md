---
layout: post
title: Implementing Autoencoders in the Fastai Library
mathjax: true
description: Step by step guide to implementing an autoencoder in fastai.
---

<figure>
  <img src="/assets/images/nn.png" alt="nn_svg" class="center"/>
  <figcaption class="center">Autoencoder Architecture. Image made using <a href="https://alexlenail.me/NN-SVG/LeNet.html">NN-SV.</a>.</figcaption>
</figure>

<div class="message">
  This post is also available on towardsdatascience <a href="https://towardsdatascience.com/autoencoders-in-the-fastai-library-fa288e1f899a">here</a>.
</div>


# Introduction
[fastai](https://docs.fast.ai/index.html) is a deep learning library that simplifies training neural networks using modern best practices [1]. While fastai provides users with a high-level neural network API, it is designed to allow researchers and users to easily mix in low-level methods while still making the overall training process as easy and accessible to all.

This post is going to cover how to set up an _autoencoder_ in fastai. This will go through creating a basic autoencoder model, setting up the data in fastai, and finally putting all this together into a learner model.

_Note: a basic understanding of fastai and PyTorch is assumed._
<br/><br/>
# Implementation
## Setting Up the Data
An autoencoder is a neural network that learns to recreate the input through some type of bottleneck in the architecture. To set this up we need a fastai databunch in which the input and output are equal.

Here I will give an example to do this with image data, however, there is a more general example to work with any array available on the corresponding notebook available [here](https://github.com/henriwoodcock/blog-post-codes/blob/master/autoencoders-in-fastai/autoencoders-in-fastai.ipynb).

The code for importing a dataset is below:
{% highlight python %}
from fastai import *
from fastai.vision import *

size = 32
batchsize = 32
tfms = get_transforms(do_flip = True)
src = (ImageImageList.from_folder(image_path).label_from_func(lambda x: x))
data = (src.transform(tfms, size=size, tfm_y=True)
        .databunch(bs=batchsize)
        .normalize(imagenet_stats, do_y = False))
{% endhighlight %}

The two differences between this and setting up a databunch for a classifier are:

{% highlight python %}
ImageImageList()
{% endhighlight %}

_ImageImageList_ is a built-in fastai list that sets both the input and output data to be an Image. Using this means we can still use __in-built fastai functions__ like show_batch.

The second difference is how the labels are set:

{% highlight python %}
label_from_func(lambda x: x)
{% endhighlight %}

This allows the user to set the labels from a defined function. Here we use a function that outputs the input.

Now with all this together, we can run:

{% highlight python %}
data
{% endhighlight %}

Which outputs:

{% highlight python %}
ImageDataBunch;

Train: LabelList (4800 items)
x: ImageImageList
Image (3, 32, 32),Image (3, 32, 32),Image (3, 32, 32),Image (3, 32, 32),Image (3, 32, 32)
y: ImageList
Image (3, 32, 32),Image (3, 32, 32),Image (3, 32, 32),Image (3, 32, 32),Image (3, 32, 32)
Path: autoencoders-in-fastai/data;

Valid: LabelList (1200 items)
x: ImageImageList
Image (3, 32, 32),Image (3, 32, 32),Image (3, 32, 32),Image (3, 32, 32),Image (3, 32, 32)
y: ImageList
Image (3, 32, 32),Image (3, 32, 32),Image (3, 32, 32),Image (3, 32, 32),Image (3, 32, 32)
Path: autoencoders-in-fastai/data;

Test: None

{% endhighlight %}

And running show_batch gives:

{% highlight python %}
data.show_batch(rows = 1)
{% endhighlight %}

<figure>
  <img src="/assets/post_images/autoencoders_in_fastai/show_batch.png" alt="show_batch" class="center"/>
  <figcaption class="center">data.show_batch(rows = 1) output. Images taken from the <a href="https://www.cs.toronto.edu/~kriz/cifar.html">CIFAR10 Dataset.</a></figcaption>
</figure>

<br/><br/>
## Creating the Model
We now need to create an autoencoder model. This is done by creating a PyTorch module. Below is a basic example taken from [here](https://github.com/jellycsc/PyTorch-CIFAR-10-autoencoder/blob/master/main.py).

{% highlight python %}
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3,12,4,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(12,24,4,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(24,48,4,stride=2,padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48,24,4,stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24,12,4,stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12,3,4,stride=2,padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x): return self.encoder(x)

    def decode(self, x): return self.decoder(x)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
{% endhighlight %}
<br/><br/>
## Creating a Fastai Learner
Now we put all this together into a fastai learner. To do this you need to define a loss function, I will be using MSE loss for this example.

To implement this we first create an instance of our autoencoder as follows:
{% highlight python %}
autoencoder = Autoencoder()
{% endhighlight %}
Then we put this into a fastai Learner:
{% highlight python %}
import torch.nn.functional as F

learn = Learner(data, autoencoder, loss_func = F.mse_loss)
{% endhighlight %}

Now we have done this we can easily utilise all the best training practices incorporated in fastai such as _lr_find_ and _fit_one_cycle_.
<br/><br/>
## Training
All the techniques implemented in fastai can now be used on your custom autoencoder.

{% highlight python %}
learn.lr_find()
learn.fit_one_cycle()
{% endhighlight %}
<br/><br/>
## Example Results
Using the fastai library I trained 10 epochs on a subset of the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) using lr_find to find an optimal learning rate and fit_one_cycle and achieved the following results:

<figure>
  <img src="/assets/post_images/autoencoders_in_fastai/training_show_results.png" alt="Training Results" class="center"/>
  <figcaption class="center">Results from 10 epochs using the fastai library.</figcaption>
</figure>
<br/><br/>
# Conclusion
Here I have presented a simple overview of how to implement an autoencoder in fastai, you can find the notebook [here](https://github.com/henriwoodcock/blog-post-codes/blob/master/autoencoders-in-fastai/autoencoders-in-fastai.ipynb) which includes all the code as well as how to implement this for a general array dataset. There is also information on how to use the encoder and decoder part once the autoencoder has been trained.

The autoencoder model I presented here was a very simple one and there are many improvements which can be made a few to mention are feature loss, upsampling instead of transposed convolutions and finally pixel shuffle. I have found these to prove very effective in some of my own work.

------

## References
[1] Jeremy Howard and Sylvain Gugger. _fastai: A Layered API for Deep Learning_. arXiv. arXiv:2002.04688 https://arxiv.org/abs/2002.04688. 2020.
