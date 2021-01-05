# How to Create a Projects Page on Jekyll
> Customise your Jekyll site to show off your portfolio.

_This tutorial is for people like myself who do not know much about Jekyll, html
or any other web framework but have a personal site. I could not find anything online
which explained how to do this simply. I hope this helps somebody like me!_

It is becoming more and more important to have an online presence. From data scientists, 
artists, writers all the way to small businesses; having a portfolio website is 
becoming critical to success.

[Jekyll](https://jekyllrb.com) is a static site generator to help create blogs or as 
[Tom Preston-Werner](https://tom.preston-werner.com) (the developer) would say to blog 
“like a hacker”. Because Jekyll powers Github pages it has become a popular solution for 
many blogs and portfolios.

This post will not go into how to create a Jekyll site as there are many tutorials 
online to do this (like [this](https://medium.com/20percentwork/creating-your-blog-for-free-using-jekyll-github-pages-dba37272730a)). However, this will be a short post on how to make an easy to use "projects" 
page (or any other page you may like) with similar functionality and ease of use as 
the blogging functionality, such as drag and dropping markdown files into a folder.

This post is split into 4 steps with a section at the end for extra features which can 
be incorporated.

## Contents

## 1. Create a Projects folder
In your Jekyll site directory, you need to create a new folder and name it

```
_your_page_name
```

For example, because mine is called projects:

```
_projects
```

![](/images/post_images/jekyll_projects_page/folder.png "My Jekyll folder structure")

## 2. Edit the \_config.yml

For Jekyll to recognise the new `_projects` folder it needs to be added to the `_config.yml`. 
This comes under the collections section of your config file. For example, add the following 
to your config:

```
collections:
  - projects
```

## 3. Create a project markdown file

Create a markdown file for one of your projects to be used as an example, this can help you 
decide on consistent front matter for your projects ready to create your website. For example, 
if I create a file called `example_project.md` in my `_projects` folder I could choose the 
following front matter:

```
---
title: Example Project
description: This is an example project.
layout: project_page
---
```

I have chosen for each of my projects to have a title and description. Also, note I 
have chosen the layout to be *project_page* which refers back to the layout I made earlier.

## 4. Create the projects web page

Now you need to create the projects web page which is what will be seen from the website. 
I want mine to be a list of projects which can be selected to open up a page containing 
the project.

![](/images/post_images/jekyll_projects_page/projects_page.png "A screenshot of my projects page.")

To do this create a new folder in the directory (I called mine `projects` for consistency) and 
create a file inside called `index.md`.

Inside `index.md` you can now structure the file to look how you like. I added `title: Projects` 
to the front matter. The main part of this is to write a loop to go through the files in your 
`_projects` folder and display them into the format you like.

For example:

```
{% for project in site.projects %}
  <h2> {{ project.title }} </h2>
  <p>{{ project.description }}</p>
{% endfor %}
```

This loop simply loops through each project in the projects folder and displays the project 
title in `h2` format and adds the project description.

## Extras

### Create a page layout

To have an individual page for each project you need to create a new layout in the 
`_layouts` directory. I called mine `project_page.html` and copied the layout for a post. 
The only variable `project_page.html` takes is a variable called `page.title` and so 
this will be required in the front matter for each project.

### Generate a web page for each project

To generate a web page for each project the `_config.yml` needs to be slightly changed 
to create an output

```
collections:
  projects:
    output: true
```

### Create a custom link to each project

This can be done for each project in the front matter, but I found it easiest to also 
specify this in the `_config.yml` by added the `permalink` key.

```
collections:
  projects:
    output: true
    permalink: /:collection/:title
```

For example, this creates a web page called `yoursite.io/projects/project_title` in my 
example.

### Create a custom link to your projects page

This can be done by editing the front matter on your `projects.md` to include the 
`permalink` key. For mine I set is as:

```
permalink: /projects/
```

## Summary

And that is it, for a live example, you can look at my portfolio website by clicking 
[here](https://henriwoodcock.github.io). You can also find all the files I created 
during this blog post on my GitHub 
[here](https://github.com/henriwoodcock/blog-post-codes/tree/master/jekyll_projects_page).

