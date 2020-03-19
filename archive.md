---
layout: page
title: Archive
---

{% for post in paginator.posts %}
  * {{ post.date | date_to_string }} &raquo; [ {{ post.title }} ]({{ post.url }})
{% endfor %}

