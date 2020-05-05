---
layout: page
title: Projects
---

{% for post in site.posts %}
  {% if post.project %}
  ### [ {{ post.title }} ]({{ post.url }})
  {% endif %}
{% endfor %}
