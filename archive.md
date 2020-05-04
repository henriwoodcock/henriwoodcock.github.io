---
layout: page
title: Archive
---

{% for post in site.posts %}
  {% if post.layout == "post" %}
    * {{ post.date | date_to_string }} &raquo; [ {{ post.title }} ]({{ post.url }})
{% endfor %}
