# Stop using numpy.random.seed() 

> How to set random seeds for individual classes in Python

Using `np.random.seed(number)` has been a best practice when using NumPy to create reproducible work. Setting the random seed means that your work is reproducible to others who use your code. But, now when you look at the Docs for `np.random.seed`, the description reads:

> This is a convenience, legacy function.
> The best practice is to not reseed a BitGenerator, rather to recreate a new one. This method is here for legacy reasons.

So what’s changed? This post will explain the old method, issues with it and then show and explain the new best practice and the benefits.

## Legacy Best Practice

If you look up tutorials using `np.random` you see many of them using `np.random.seed` to set the seed for reproducible work. We can see how this works:

```python
>>> import numpy as np

>>> np.random.rand(4)
array([0.96176779, 0.7088082 , 0.06416725, 0.82679036])

>>> np.random.rand(4)
array([0.15051909, 0.77788803, 0.67073372, 0.32134285])
```

As you can see, two calls to the function lead to two completely different answers. If you want somebody to be able to reproduce your projects, you can set the seed. You can do this with the following code snippet:

```python
>>> np.random.seed(2021)
>>> np.random.rand(4)
array([0.60597828, 0.73336936, 0.13894716, 0.31267308])

>>> np.random.seed(2021)
>>> np.random.rand(4)
array([0.60597828, 0.73336936, 0.13894716, 0.31267308])
```

Now, you see the results are the same. If you need to prove this to yourself, you can enter the above code on your Python setup.

Setting the seed means the next random call is the same; it sets the sequence of random numbers such that any code that produces or uses random numbers (with NumPy) will now produce the same sequence of numbers. For example, look at the following:

```python
>>> np.random.seed(2021)
>>> np.random.rand(4)
array([0.60597828, 0.73336936, 0.13894716, 0.31267308])
>>> np.random.rand(4)
array([0.99724328, 0.12816238, 0.17899311, 0.75292543])
>>> np.random.rand(4)
array([0.66216051, 0.78431013, 0.0968944 , 0.05857129])
>>> np.random.rand(4)
array([0.96239599, 0.61655744, 0.08662996, 0.56127236])

>>> np.random.seed(2021)
>>> np.random.rand(4)
array([0.60597828, 0.73336936, 0.13894716, 0.31267308])
>>> np.random.rand(4)
array([0.99724328, 0.12816238, 0.17899311, 0.75292543])
>>> np.random.rand(4)
array([0.66216051, 0.78431013, 0.0968944 , 0.05857129])
>>> np.random.rand(4)
array([0.96239599, 0.61655744, 0.08662996, 0.56127236])
```

## So what’s the problem?

You may be looking at the above example and be thinking, “so what’s the problem?”. You can create reproducible calls, which means that all random numbers generated after setting the seed will be the same on any machine. For the most part, this is true; and for many projects, you may not need to worry about this.

The problem comes in larger projects or projects with imports which could also set the seed; using `np.random.seed(number)` sets what NumPy calls the global random seed — affecting all uses to the `np.random.*` module. Some imported packages or other scripts could reset the global random seed to another random seed with `np.random.seed(another_number)`, leading to undesirable changes to your output and your results not being reproducible. For the most part, you will only need to ensure that you use the same random numbers for specific parts of your code (like tests or functions).

## The solution and new method

For the reason above (and many others), NumPy has moved towards advising users to create a random number generator for specific tasks or to even pass around when you need parts to be reproducible.

> “The preferred best practice for getting reproducible pseudorandom numbers is to instantiate a generator object with a seed and pass it around” — Robert Kern, NEP19.

Using this new best practice looks like the following:

```python
import numpy as np

>>> rng = np.random.default_rng(2021)
>>> rng.random(4)
array([0.75694783, 0.94138187, 0.59246304, 0.31884171])
```

As you can see, these numbers are different from the earlier example, that is because NumPy has changed the default pseudo-random number generator. However, the old results can be replicated by using `RandomState` which is a generator for old legacy methods.

```python
>>> rng = np.random.RandomState(2021)
>>> rng.rand(4)
array([0.60597828, 0.73336936, 0.13894716, 0.31267308])
```

## The benefits

You can pass Random number generators around between functions and classes, meaning each individual or function could have its own random state without resetting the global seed or each script could pass a random number generator to functions that need to be reproducible. The benefit is you know _exactly_ what random number generator is used in each part of your project.

```python
def f(x, rng): return rng.random(1)

#Intialise a random number generator
rng = np.random.default_rng(2021)

#pass the rng to functions which you would like to use it
random_number = f(x, rng)
```

Other benefits arise with parallel processing. Albert Thomas has a great blog post on this [here](https://albertcthomas.github.io/good-practices-random-number-generators/).

## Conclusion

Using independent random number generators can help improve the reproducibility of your results, this is done by not relying on the global random state (which can be reset or used without knowing). Passing around a random number generator means you can keep track of _when_ and _how_ it was used and ensure your results are the same.

## Sources and More Information

- [NumPy Random Seed, Explained](https://www.sharpsightlabs.com/blog/numpy-random-seed/), Joshua Ebner, 2019.
- [Good practices with numpy random number generators](https://albertcthomas.github.io/good-practices-random-number-generators/), Albert Thomas, 2020.
- [NEP19 — Random Number Generator Policy](https://numpy.org/neps/nep-0019-rng-policy.html), Robert Kern, 2018.
