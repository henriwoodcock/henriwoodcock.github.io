---
layout: page
title: Projects
liner: My personal projects 
---

{% for post in site.posts %}
  {% if post.project %}
  <h3>
    <a href="{{ site.baseurl }}{{ post.url }}">
      {{ post.title }} </a>
  </h3>
  <small> {{ post.description }} </small>
  {% endif %}
{% endfor %}
