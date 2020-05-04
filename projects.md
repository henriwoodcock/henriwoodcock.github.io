---
layout: page
title: Projects
---

{% for project in site.posts %}
  * [ {{ project.title }} ]({{ project.url }})
{% endfor %}
