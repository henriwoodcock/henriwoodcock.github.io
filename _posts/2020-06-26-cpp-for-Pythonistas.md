# C++ for Pythonistas and Data Scientists
> An introduction to Cling to help learn C++

<br>

{% include info.html text="This post is also available on towardsdatascience [here](https://towardsdatascience.com/c-for-pythonistas-and-data-scientists-2e1a74a7b8be)" %}

<br>

When studying mathematics at University I was introduced to both Python and R. Since then I have only stuck to those two languages and dabbled in other languages when needed.

Recently, I have wanted to improve my programming foundations and learn more about the underlying concepts that we data scientists take for granted with Python and R, while also finding some ways to improve my workflows. And so I undertook the challenge of learning C++, doing this then led me to find [Cling](https://root.cern.ch/cling).


<br>


## Cling
[Cling](https://root.cern.ch/cling) is an interactive interpreter for C++ which helps give a similar experience to coding with Python. This has had a huge benefit to my learning experience with C++ and I am sure it can help many others.


<br>


## Hello, World!
A basic C++ program for Hello, World! can be daunting and off putting for a Python user. Below is an example:

```cpp
#include <iostream>

int main() {
 std::cout << "Hello, World!" << std::endl;
 return 0;
}
```

Once written, the program can then be compiled (if there are no errors) and then ran to see the output.


Below is the same but when using the Cling interpreter:


```cpp
#include <iostream>
cout << "Hello, World"! << endl;
```

No compiling is needed!


This is a very simple example, but using Cling can allow you to explore C++ functions, vectors and more in an interactive way allowing you to experiment more .


For example, to experiment with vectors you can use the interpreter to help learn the functions and syntax.


```cpp
#include <vector>
vector<int> data;


data.push_back(1);
data.push_back(2)


// return the vector to see what happened
data;
```

<br>


## Jupyter Notebooks
There is also a Jupyter Kernel which can make learning even more interactive while also helping you learn ways you can present your work in a data science flow. you'll start doing some heavy data processing in C++.


![](/images/post_images/cpp_for_data_scientists/screenshot2.png "Example Notebook.")

<br>


## Installation
Installing is easy and can be done with [Conda](https://anaconda.org/anaconda/conda):

```bash
conda install -c conda-forge cling
```

To get the Jupyter Kernel also install `xeus-cling`:
```bash
conda install xeus-cling -c conda-forge
```

### Setting Up Jupyter Notebooks
Run `Jupyter Notebook` as normal and then select a C++ kernel as seen in the screenshot below:

![](/images/post_images/cpp_for_data_scientists/screenshot1.png "Creating a C++ Kernel.")

<br>


## Final Remarks
Interpreting C++ is not a final solution and using C++ like this means you lose a lot of the power possible with the language. But, as a learning tool Cling is invaluable and can help speed up your learning process if you already have some foundations in other languages.

The associated code and notebook are available on my [GitHub](https://github.com/henriwoodcock/blog-post-codes), this also includes code to import a simple single columned CSV.
