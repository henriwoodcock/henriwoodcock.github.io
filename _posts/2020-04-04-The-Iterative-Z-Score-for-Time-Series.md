---
layout: post
title: Iterative Z-Score for Time Series
mathjax: true
---

__random sampling is like eating soup.__ -> with a good stir

_one:_
__Statistical inference__ is a subfield of statistics where we make conclusions about a _population_ by analysing _sample data_.

_two:__
Often in statistics we make conclusions about a _population_ from analysis of _sample data_. This is called __statistical inference__.

_one:_
We define population as an entire set of measurements or counts we wish to make conclusions, for example if we wanted to analyse the heights of university professors, the population would be the heights of all the professors in the world. On the other hand, a sample is a subset of the population, in the same example a sample could be the heights of professors from just one country.

_two:_
A population is the whole set of measurements or counts we wish to make conclusions, this could be the heights of all professors in the world. A sample is a subset of the population, in this example a sample could be the heights of professors from one institution.

_one:_
This post will be concerned with sample data taken from _random sampling_, that is when each count or measurement of a population is selected at random.

_two:_
This post will be concerned with _random sampling_, that is when each count or measurement of a population is selected randomly. There are many benefits to random sampling which will not be discussed here, but some of them can be found here[insert link here].

# Why is Sampling Important?
# Why we need to sample
# Aim of sample helps to pick how wide to view sampling. Need to understand population
-- explain what p hat is
_no ->_ Some of the reasons for sampling are:  _(how to make this sentence better?)_

- Saving time and money: when taking a survey it is not practicable to ask the entire population to take part in the survey (although many call centres will try)

- Inferences can still be made from sample data about the population, provided a good sample number and sampling method

- In some situations there is no choice but to sample, for example checking if every firework in a box of fireworks worked would mean there are no fireworks left to use at the bonfire _-> makes sense?_

# Confidence Intervals
This method of sampling makes use of confidence intervals for a proportion. You can found out more about confidence intervals from an earlier post[insert here]. However for this post we just need the formula and definition.

A _confidence interval_ is an estimate of plausible values for an unknown parameter. Often statisticians use a 95% confidence level. A 95% confidence interval means that 95 of 100 confidence intervals calculated from sample data will contain the true parameter.

A confidence interval for a proportion from sample data with a sample size of $n$ can be calculated as follows:

$$ \hat{p} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} $$.

Where z is confidence level from the normal distribution. For a 95% confidence interval this becomes:

$$ \hat{p} \pm 1.96 \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} $$.


# Sample Size Determination
The minimum required number of samples to best represent a population can be found through rearranging confidence intervals. You may have seen this in a real world survey when the results have been quoted as "within an error term of 0.02 at a 95% confidence level".

You select a width of the confidence interval such that out of 95% of samples, the true proportion is represented to an accuracy of this width.

Call this width x, we thus have:

$$z z \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} \leq x$$.

Rearranging for n we get:

$$ n \geq z^2 \frac{\hat{p}(1-\hat{p})}{x^2} $$

Therefore to calculate the proportion of a binary measurement in a population we would require n samples to be taken.

# Conclusion
This post has helped give an introduction to finding the optimal number of samples required to be taken to calculate the proportion of measurements to a confidence level and a degree of accuracy.

When I first looked into this problem I struggled to find many resources online, so I hope this can help many people especially for people doing their dissertations/thesis (in social sciences) in which taking interviews and surveys are required.

Another post will be made which extends this theory to the _multinomial case_, that is when options to a question are greater than $2$.
