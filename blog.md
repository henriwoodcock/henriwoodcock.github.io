---
layout: page
title: Blog
---

{% for post in site.posts %}
  {% if post.project == null %}
  <h3><a href="{{ post.url }}">"{{ post.title }}"</a></h3> <span class="post-date"> <small><small>{{ post.date | date_to_string }}</small></small></span>
  ### [ {{ post.title }} ]({{ post.url }}) {{ post.date | date_to_string }}
  <span> A little description of the post </span>
  <span class="post-date"> <small><small>{{ post.date | date_to_string }}</small></small></span>
  {% endif %}
{% endfor %}
