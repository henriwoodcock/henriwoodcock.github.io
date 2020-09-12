---
layout: post
title: Optimal Sample Size
mathjax: true
description: How to statistically calculate the optimal sample size in a binary distribution.
---

# Introduction
Finding the optimal sample size can be important for many different contexts, from collecting voting intentions in an election to assessing the quality of machinery in a company. Finding a sample which best represents a population can help reduce costs and time while also providing conclusions which can be applied to the entirety of the population.

A way to help understand this is from George Gallup who was a pioneer of survey sampling techniques.

>"If you have cooked a large pan of soup, you do not need to eat it all to find out if it needs more seasoning. You can just taste a spoonful, provided you have given it a good stir", George Gallup.

To select a sample size, methods from statistics can be used. This can allow _quantitative_ measures of our confidence in our samples.

This post is going to first give a mathematical proof for the selection of sample size and then do a computational proof in Python to show how this works on some simulated data.

# Strategy
This section will discuss how to find the optimal sample size in a binary situation - this is when there are two outcomes. An example could be a poll with a yes or no question or a good or bad rating of machinery.

First, this post will go through confidence intervals which is what this method is based on and then finally go onto how to calculate the optimal sample size from a confidence interval.

## Setup
In statistics, a _sample_ is a set of objects or measurements which is a subset of the entire population taken through a statistical sampling technique. Let's call the two outcomes for the binary problem _outcome 1_ and _outcome 2_.
Define $p$ as the probability of outcome 1 occurring, giving the probability of outcome 2 as $1-p$.
The question is, _what sample size will we need so we can calculate $p$ to a high accuracy?_

## Confidence Intervals
To answer this we need a confidence interval for $p$.

__Definition__: a $95%$ confidence interval for $p$ is an interval in which if we calculated the interval $100$ times, $95$ of them would contain the true value of $p$ for a given sample size $n$.

For the general case, the formula for this is:
$$ \hat{p} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

Where $\hat{p} is our estimate for $p$ (what we are trying to find), $z$ represents a z-score from the normal distribution (for example 1.96 would relate to a 95% z-score).

## Optimal Sample Size
Now to calculate the optimal sample size we can look at it the other way and rearrange this formula to find $n$. First, let's expand the confidence interval:
$$\hat{p} \in [-z\sqrt{\hat{p}(1-\hat{p})/n} , z\sqrt{\hat{p}(1-\hat{p})/n} ]$$

Because we want to find $p$ to a high accuracy we will want to make this confidence interval smaller than a certain width (of your choice), let's call this width, $W$. Putting this together we get:

$$z\sqrt{\hat{p}(1-\hat{p})/n} - (-z\sqrt{\hat{p}(1-\hat{p})/n}) < W$$

Which gives:

$$
2z\sqrt{\hat{p}(1-\hat{p})/n}< W
\implies z\sqrt{\hat{p}(1-\hat{p})/n}< W/2
$$

and now continue rearrange this until $n$ is on one side:

$$
\sqrt{\hat{p}(1-\hat{p})/n}< \frac{W}{2z}
\frac{\hat{p}(1-\hat{p})}{n}< \frac{W^2}{4z^2}
n > \frac{4z^2\hat{p}(1-\hat{p})}{W^2}
$$

This is a general formula that can be used if you already have a good estimate for $p$, however if not, this can be taken even further with an assumption. Let's look at the formula for the variance of a binomial distribution:
$$\sigma = np(1-p)$$

To encode our uncertainty we can maximise this variance, with $n$ held constant this is achieved by assuming a $p$ value of $0.5$. This now gives us:
$$n > \frac{4z^2(0.5)(1-0.5)}{W^2}$$

Which can be simplified to become:
$$n > \frac{z^2}{W^2}$$

Now, let's input some values. For a $95%$ confidence level, we want to set $z = 1.96$ (from the standard normal distribution) and let $W = 0.02$. $W$ is the width of the confidence interval which means $W/2$ is the margin of error, so for this example I have a margin of error of $0.01$.
Putting all this together:
$$ n > \frac{1.96^2}{(2\times 0.01)^2} $$

>For this example we can now say, _95% of samples of sample size greater than 9604 will contain the true proportion within an accuracy of 1%_.

# Example
Now the mathematical proof is done I will generate some data to give an hands-on example.

<script src="https://gist.github.com/henriwoodcock/2f36eb31a6b21871cad3838ab20b032d.js"></script>

# Further Reading
This post assumes the total population is unknown and infinite. If the total population is known there are ways to account for this: https://byjus.com/sample-size-formula/
For more on confidence intervals here is a good introduction https://towardsdatascience.com/understanding-confidence-interval-d7b5aa68e3b
For more on the importance of sample size check out: https://medium.com/swlh/is-the-sample-size-of-a-poll-important-2b25b5bfe64d

# Conclusion
This post has presented a method to find a good sample size for a given confidence level and allowed margin of error, however only the binary method has been covered here. To extend the results calculated here to the multiclass scenario (where there are more than two options), the problem can be simply treated as several binary problems. This is left to the reader.
For the associated formula notebook and the example notebook please go to my [GitHub](https://github.com/henriwoodcock/blog-post-codes/tree/master/optimal_sample_size).
