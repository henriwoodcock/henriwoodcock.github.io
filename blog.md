---
layout: page
title: Blog
---

{% for post in site.posts %}
  {% if post.project == null %}
  ### [ {{ post.title }} ]({{ post.url }})
  <span> A little description of the post </span>
  <span class="post-date"><small>{{ post.date | date_to_string }}</small></span>
  {% endif %}
{% endfor %}
