---
layout: page
title: Blog
---

{% for post in site.posts %}
  {% if post.project == null %}
  ### [ {{ post.title }} ]({{ post.url }})
  A little description of the post
  {{ post.date | date_to_string }}
  {% endif %}
{% endfor %}
