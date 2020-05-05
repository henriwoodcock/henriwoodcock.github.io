---
layout: page
title: Blog
---

{% for post in site.posts %}
  {% if post.project == null %}
  ### [ {{ post.title }} ]({{ post.url }})
  A little description of the post
  <span class="post-date"><small>{{ page.date | date_to_string }}</small></span>
  {% endif %}
{% endfor %}
