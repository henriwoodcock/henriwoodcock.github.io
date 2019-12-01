---
layout: post
title: Calculating The Minimum Distance Between Points and Lines
mathjax: true
---

 Recently I had to find the distance of a point to the closest road. The dataset consisted of just over 100000 points and over 10 million lines.

## Mathematics
I am sure all of you already know how to calculate the distance between a point and a line, however I am including this to help keep the intuition in our minds as it can be confusing to get straight into the code.

Given two points $P_{1} = (x_{1}, y_{1})$ and $P_{2} = (x_{2}, y_{2})$ which form a line, and a third point $P_{3} = (x_{3}, y_{3})$ which is another point (which could be on the line), the minimum distance from the point to the line is calculated by:

$$d = \frac{|(y_{2} - y_{1})x_{3} - (x_{2} - x_{1})y_{3} + x_{2}y_{1} - x_{1}y_{2}|}{\sqrt{(y_{2} - y_{1})^2 + (x_{2} - x_{1})^2}}$$

Or with the equation of the line $P_{1} \rightarrow P_{2}$ as $ax + by + c = 0$, it can be calculated by:

$$d = \frac{|ax_{3} + by_{3} + c|}{\sqrt{a^2 + b^2}}$$

<div class="message">
  The proof is not shown here, however if the reader is struggling with the intuition it can be proven quickly by themself. Hint: find a line perpendicular to the line $P_{1} \rightarrow P_{2}$ that goes through the point $P_{3}$.
</div>

# Numerical Example
![without point on line](/assets/post_images/minimum_dist_to_line_post/plot_without_closest_point.png)
In this example, let $P_{1} = (1,6), P_{2} = (13, -4), P_{3} = (6,5)$. Therefore,
$$ d = \frac{|(-4 - 6)6 - (13 - 1)5 + (13)(6) - (1)(-4)|}{\sqrt{-4-6)^2 + (13-1)^2}} \approx 2.43 (3sf)$$

This will calculate the distance of the line:
![with point on line](/assets/post_images/minimum_dist_to_line_post/plot_with_closest_point.png)
