---
layout: page
title: Projects
liner: My personal projects 
---

{% for post in site.posts %}
  {% if post.project %}
  ### [ {{ post.title }} ]({{ post.url }})
  <small> {{ post.description }} </small>
  {% endif %}
{% endfor %}
