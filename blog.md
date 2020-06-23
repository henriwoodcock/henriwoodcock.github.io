---
layout: page
title: Blog
liner: I like to write about Data Science to coding.
---

{% for post in site.posts %}
  {% if post.project == null %}
  <h3>
    <a href="{{ site.baseurl }}{{ post.url }}">
      {{ post.title }} </a>
  </h3>
  <span> {{ post.description }} </span>
  <span class="post-date"> <small><small>{{ post.date | date_to_string }}</small></small></span>
  {% endif %}
{% endfor %}
