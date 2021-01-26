# A Simple Guide to Creating an Image Dataset

> A full end to end tutorial to create a dataset without web scraping!

![](images/post_images/img_dataset/pexels-taryn-elliott-4340919.jpg "Photo by Taryn Elliott from Pexels")

Data scientists love creating models and competing to slightly improve accuracy
on datasets. By now, so many data scientists have used these popular datasets
that it has become difficult to learn anything new from them. It also stops
data science from being a problem solving subject to a more engineering
subject: slightly tweaking models to slightly improve scores.

To do something unique in data science, you will have to create a dataset
yourself and solve a new problem! Because most of us data scientists do not
know much about data engineering or web scraping, this guide will show you how
you can make an image dataset for your unique problem without the need to know
anything about web scraping! This post will cover creating an image dataset
for a classification problem and break it down into the following steps:

1. Defining a problem
2. Data collection
3. Cleaning the dataset

## Defining a problem

The first part of making a new dataset is deciding on your problem. You cannot
just collect data and hope you can figure something out. There is too much data
out there for that! To solve an image classification problem, you need to
decide the classes you want to classify. For example, is it dog breeds? Or the
artist who produced paintings? When I created a dataset, I knew I wanted to
classify flood defense assets from Images, and so I started to list out
different assets to include in my dataset.

With this decision, you can list out the different classes you want to classify.
For my dataset, this was: embankments, flood gates, floodwalls, outfalls,
reservoirs, and weirs. However, you can choose as many or as few as you like!

## Data Collection

For the data collection part, we can use google to find the images we like. I
promised no web scraping knowledge is required, meaning this method will be as
simple as possible. You will not even need a web scraping package!

Here are the steps:

1. Go to Google Images
2. Search your keyword. For example “Flood Gate”
3. Go to the very bottom of the page and press the “load more” button
4. Keep repeating step 3 until the load more button not longer appears
5. Open the console on your web browser and enter:

    ```
    var urls = Array.from(document.querySelectorAll(‘.rg_i’)).map(el=> el.hasAttribute(‘data-src’)?el.getAttribute(‘data-src’):el.getAttribute(‘data-iurl’));
    var filtered = urls.filter(function (el) {return el != null;});
    var hiddenElement = document.createElement(‘a’);
    hiddenElement.href = ‘data:text/csv;charset=utf-8,’ + encodeURI(filtered.join(‘\n’));
    hiddenElement.download = ‘file.csv’; hiddenElement.click();
    ```

Hitting enter will generate a CSV file with links to all the images in your
google search. You can now rename this CSV to be the name of the class you just
searched. Repeat this for each category you wish to include.

![](images/post_images/img_dataset/img-of-google-search.png "The console on Safari.")

The next step to data collection is downloading the images in the CSV. We can
do this by opening each CSV and then looping through all the rows to download
each image. For this step, I used `csv` and `requests`. First install
`requests` with `pip` (`csv` comes pre-installed with Python3):

```
pip install --upgrade pip
pip install requests
```

The following is some pseudo-code to loop through the CSV files and download
all the images. You can find a full example of the code I used with all the
whistles and bells on my
[Github](https://github.com/henriwoodcock/automatic-asset-classification/blob/master/automatic_asset_classification/web_scrape/web_scraping.py).

```
from pathlib import Path
import requests
import csv

dataLoc = Path('data') #location where data is saved
csvLoc = dataLoc / 'csvs' # folder contained csvs in data folder

classes = ["class1", "class2"] #class names

for class in classes:

  #csv files are named by class.csv
  with open(csvLoc / (str(class) + "_image_db.csv")) as csvfile:
    #create a list of rows
    csvrows = csv.reader(csvfile, delimiter=',', quotechar='"')

  #create a folder to output images
  classLoc = dataLoc / class

  #if the folder does not exist make it
  if not classLoc.exists(): classLoc.mkdir()

  #number each image by number in csv
  i = 0
  for row in csvsrows:
    #imageurl is first (and only column)
    imageurl = row[0]

    #imagename = "class_i.jpg"
    imagename = class + str(i) + ".jpg"

    result = requests.get(url, stream = True)
    image = result.raw.read()

    #save image into folder for that class
    open(classLoc / imagename, "wb").write(image)
```

Doing this may take some time (depending on your internet speed and the number
of images to be downloaded). So feel free to leave this running in the
background.

Once done, if using a similar download format to me above you should have a
folder containing folders for each class with images of that class inside.

## Cleaning the Dataset

Downloading the data from Google means we need to do some final checks. Doing
this makes sure the data is to a high standard. Cleaning the images ready for
use consists of three parts:

1. Removing duplicate images.
2. Removing incorrectly labeled images (from the web scrape).
3. Cropping images.

### 1. Removing Duplicate Images

There are many different algorithms to remove duplicate images. I will now
explain a few of them. The first check is to verify that there are no corrupted
images (this could be from failed file downloads). We can do this by writing a
loop in Python that attempts to open each image. In Python, this can be done
with a `try` and `except` block. If we are unable to open the image, delete the
file.

After removing the broken images, we can start deleting duplicate files. We
will now look at two methods to do this. (Again, the code for this can be found
on my Github
[here](https://github.com/henriwoodcock/automatic-asset-classification/blob/master/automatic_asset_classification/web_scrape/hashing_functions.py) and
[here](https://github.com/henriwoodcock/automatic-asset-classification/blob/master/automatic_asset_classification/web_scrape/duplicate_images.p)).

The first method involves using a _hash function_. A hash function maps data
from an arbitrary size to a fixed size. The benefit to a hash function is that
it is computationally infeasible to find two inputs that map to the same output
(and thus unlikely to happen). We can exploit this by checking if any two of
our images have the same _hash value_. To do this import the images into
NumPy arrays and compare each image with every other image by comparing the
result of passing them through a hash function. For this, I used the md5 hash
function. Below is some pseudo-code to get the basic idea of this:

```
import hashlib
hash_keys = {}

with open(imagename, 'rb') as f:
  #produce hash of file
  imagehash = hashlib.md5(f.read()).hexdigest()

#check if file already in hash_keys
if imagehash not in hash_keys:
  hash_keys[filehash] = imagename
  #if file already in hash_keys add to duplicates
else:
  duplicates.append(imagename)
```

The second method finds images that are similar but not just exact copies. We
can use the _Hamming distance_ to calculate the similarity between the two
images. The Hamming distance is equal to the number of differences between the
two images. Because it counts the differences, images will have to be resized
such that they are of the same dimensions. We can then remove images based on a
threshold, such as if 10% or more values are different. Below you can see an
example of the Hamming distance:

```
x = [1,5,6,9]
y = [2,5,6,2]
hamming_distance(x,y) = |1-2| + |5-5| + |6-6| + |9-2|
                      = 1+0+0+1
                      = 2
percentage_difference = hamming_distance(x,y) / len(x)
                      = 2 / 4
                      = 0.5
```

### 2. Removing Incorrect Images

Removing incorrect images was done in a semi-automated fashion. For each
category, take the first 50 images and label them “yes” (if the image is of the
category) and “no” (if the image is not of the category). Now you fine-tune a
pretrained neural network to predict “yes” or “no” for the rest of the images
in that category. If the model predicted “yes”, you keep the image and if “no”
you remove the image. An example of doing this with Fastai can be seen on my
[Github](https://github.com/henriwoodcock/automatic-asset-classification/blob/master/automatic_asset_classification/web_scrape/image_processing.py).

![](images/post_images/img_dataset/yes_no_folder.png "An example of the yes/no folder for Flood Gate.")

### 3. Cropping Images ready for use

After all the above processing, the dataset left is almost in its final form.
The last step to prepare the data ready for use is to crop the images so that
the object is centred is and so that all images are square. This is all done by
hand to avoid issues with automated algorithms putting the wrong object center,
however you can also use an automated algorithm if you have too many images to
go through. While doing this, images can be checked if they were correctly
labelled too.

This is done to make sure that when you develop a model on your dataset, it
only learns features from the object in interest. This is especially important
in case an image contains multiple objects, you would not want it to learn the
wrong label. Images are squared as it means that many pretrained architectures
can be used and it has become a standard in deep learning.

## Summary

You have now made your Image dataset without any prior knowledge of web
scraping! You can now use it in your projects!

I will quickly make a few points about the positives and negatives of this
method.

Firstly, you do not need any web scraping software or packages. You can
download thousands of images related to your Google search, meaning you spend
less time writing scripts and more time using your new dataset.

However, there is a lot of manual work; you have to Google search for each
category and manually scroll down until there are no more images. You are also
limited to Google search and sometimes you may know of better websites which
are more specifically related to your topic. To use these websites you would
need to write a specific script for the website.

Overall this method provides you with an easy to use solution to help you
develop your image dataset. Using a method like this means you can soon get up
and running with developing your models and testing the feasibility of solving
the problem.

__Disclaimer__: Images found on Google fall under many different copyright
licences. It is important to check the law. The following use cases are exempt:
non-commercial research and private study, quotation, news reporting, education,
and some other uses. If you wish to use the data in any other way (such as
publishing online) you need to ask permission from each copyright owner.

### References

1. Nayak, P. Create an image dataset from Google Images and classify the images
using Fast.ai. _Medium_. 2020.

2. Menezes, A. J. (Alfred J.). Handbook of applied cryptography.
_Boca Raton: CRC Pre._ 1997.

3. Hill, R. A First Course In Coding Theory (Oxford Applied Mathematics And
Computing Science Series). _Oxford University Press, U.S.A_. 1986.
