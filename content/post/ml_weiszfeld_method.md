---
title: "Weiszfeld Algorithm"
date: "2021-03-12"
tags: [machine-learning, weiszfeld_algorithm]
draft: false
---

Pierre de Fermat 1607-1665 was a French lawyer and mathematician.  In 1640, he proposed a problem to Evangelista Torricelli, a student of the famous Galileo Galilei. Fermat challenged Torricelli to find the point in a triangle whose sum of  distances from the vertices is a minimum. Torricelli did solve the problem, in more than a single way, but over the years other solutions where found. In 1937, Endre Weitzfeld came up with an algorithmic solution of this problem, that we shall look into in this post.

The problem of finding the optimal placement of facilities to minimize costs is an application go this algorithm. It is also used in some form in cluster analysis in order to find the centroid of a cluster.

Assuming that all locations have the same weight, we start with finding the total cost of a system where we want to implement a single location scenario,
$$Z = \sum_{i=1}^{m} \sqrt{(x_i -c_x)^2+(y_i -c_y)^2}$$

Differentiating with respect to c_x

$$\frac{\partial Z}{\partial c_x} = \sum_{i=1}^{m} \left(-\frac{1}{2}  \frac{1}{ \sqrt{(x_i -c_x)^2+(y_i -c_y)^2}} \right)  2(x_i - c_x)   (-1)$$

$$ = \sum_{i=1}^{m} \frac{(x_i - c_x)}{ \sqrt{(x_i -c_x)^2+(y_i -c_y)^2} }$$

at the minimum value this value will be equal to zero.

$$ \sum_{i=1}^{m} \frac{(x_i - c_x)}{ \sqrt{(x_i -c_x)^2+(y_i -c_y)^2} } = 0$$

$$c_x = \frac{\sum_{i=1}^{m} \frac{x_i}{\sqrt{(x_i -c_x)^2+(y_i -c_y)^2}}}{\sum_{i=1}^{m} \frac{1}{\sqrt{(x_i -c_x)^2+(y_i -c_y)^2}}}$$

An therefore we can deduce;

$$c_{x}^{k+1} = \frac{\sum_{i=1}^{m} \frac{x_i}{\sqrt{(x_i -c_{x}^{k})^2+(y_i -c_y)^2}}}{\sum_{i=1}^{m} \frac{1}{\sqrt{(x_i -c_{x}^{k})^2+(y_i -c_y)^2}}}$$

similarly

$$c_{y}^{k+1} = \frac{\sum_{i=1}^{m} \frac{y_i}{\sqrt{(x_i -c_x)^2+(y_i -c_{y}^{k})^2}}}{\sum_{i=1}^{m} \frac{1}{\sqrt{(x_i -c_x)^2+(y_i -c_{y}^{k})^2}}}$$

if we define $$\delta_x = |c^{k+1}_x - c^{k}_x| $$

and $$\delta_y = |c^{k+1}_y - c^{k}_y| $$

Stop when $$\delta_x \leq \xi$$ and
$$\delta_y \leq \xi$$

Otherwise repeat

The attached [Excel](/post/files/weiszfeld_algorithm.xlsx) file illustrates the execution of this algorithm for three and four points.
