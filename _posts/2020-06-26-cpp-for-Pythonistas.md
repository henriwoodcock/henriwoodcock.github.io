---
layout: post
title: C++ for Pythonistas and Data Scientists
mathjax: true
description: Setting up Cling to help learn C++.
---
> An introduction to Cling to help learn C++

<br>

<div class="message">
  This post is also available on towardsdatascience <a href="https://towardsdatascience.com/c-for-pythonistas-and-data-scientists-2e1a74a7b8be">here</a>.
</div>

<br>

When studying mathematics at University I was introduced to both Python and R. Since then I have only stuck to those two languages and dabbled in other languages when needed.

Recently, I have wanted to improve my programming foundations and learn more about the underlying concepts that we data scientists take for granted with Python and R, while also finding some ways to improve my workflows. And so I undertook the challenge of learning C++, doing this then led me to find [Cling](https://root.cern.ch/cling).


<br>


## Cling
[Cling](https://root.cern.ch/cling) is an interactive interpreter for C++ which helps give a similar experience to coding with Python. This has had a huge benefit to my learning experience with C++ and I am sure it can help many others.


<br>


## Hello, World!
A basic C++ program for Hello, World! can be daunting and off putting for a Python user. Below is an example:

{% highlight cpp %}
#include <iostream>
#include <vector>


int main() {
 std::cout << "Hello, World!" << std::endl;
 return 0;
}
{% endhighlight %}


Once written, the program can then be compiled (if there are no errors) and then ran to see the output.


Below is the same but when using the Cling interpreter:


{% highlight cpp %}
#include <iostream>
cout << "Hello, World"! << endl;
{% endhighlight %}


No compiling is needed!


This is a very simple example, but using Cling can allow you to explore C++ functions, vectors and more in an interactive way allowing you to experiment more .


For example, to experiment with vectors you can use the interpreter to help learn the functions and syntax.


{% highlight cpp %}
#include <vector>
vector<int> data;


data.push_back(1);
data.push_back(2)


// return the vector to see what happened
data;
{% endhighlight %}


<br>


## Jupyter Notebooks
There is also a Jupyter Kernel which can make learning even more interactive while also helping you learn ways you can present your work in a data science flow. you'll start doing some heavy data processing in C++.


<figure>
 <img src="/assets/post_images/cpp_for_data_scientists/screenshot2.png" alt="Jupyter_Screenshot2" class="center"/>
 <figcaption class="center">Example Notebook.</figcaption>
</figure>


<br>


## Installation
Installing is easy and can be done with [Conda](https://anaconda.org/anaconda/conda):
{% highlight bash %}
conda install -c conda-forge cling
{% endhighlight %}


To get the Jupyter Kernel also install `xeus-cling`:
{% highlight bash %}
conda install xeus-cling -c conda-forge
{% endhighlight %}


### Setting Up Jupyter Notebooks
Run `Jupyter Notebook` as normal and then select a C++ kernel as seen in the screenshot below:

<figure>
 <img src="/assets/post_images/cpp_for_data_scientists/screenshot1.png" alt="Jupyter_Screenshot" class="center"/>
 <figcaption class="center">Creating a C++ Kernel.</figcaption>
</figure>


<br>


## Final Remarks
Interpreting C++ is not a final solution and using C++ like this means you lose a lot of the power possible with the language. But, as a learning tool Cling is invaluable and can help speed up your learning process if you already have some foundations in other languages.

The associated code and notebook are available on my [GitHub](https://github.com/henriwoodcock/blog-post-codes), this also includes code to import a simple single columned CSV.
