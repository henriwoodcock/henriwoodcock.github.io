---
layout: page
title: Projects
---

{% for post in site.projects %}
  * {{ post.date | date_to_string }} &raquo; [ {{ post.title }} ]({{ post.url }})
{% endfor %}
