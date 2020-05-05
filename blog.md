---
layout: page
title: Blog
---

{% for post in site.posts %}
  {% if post.project == null %}
  ### [ {{ post.title }} ]({{ post.url }})
  <span> A little description of the post </span>
  <span class="post-date">{{ post.date | date_to_string }}</span>
  {% endif %}
{% endfor %}
