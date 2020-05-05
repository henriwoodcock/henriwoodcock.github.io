---
layout: page
title: Blog
---

{% for post in site.posts %}
  {% if post.project == null %}
  <h3 style="padding:0px;margin:0px"> [ {{ post.title }} ]({{ post.url }}) </h3>
  <span> A little description of the post </span>
  <span class="post-date"><small>{{ post.date | date_to_string }}</small></span>
  {% endif %}
{% endfor %}
