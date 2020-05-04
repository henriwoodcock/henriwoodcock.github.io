---
layout: page
title: Projects
---

{% for post in site.projects %}
  * [ {{ post.title }} ]({{ post.url }})
{% endfor %}
