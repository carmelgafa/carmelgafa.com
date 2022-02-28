---
title: "Analyzing the various Gradient Descent Algorithms"
date: 2022-02-25
tags: []
draft: true
---

In this [series of posts](/tags/linear-regression/) we have discussed the basics of linear regression and they introduced the gradient descent algorithm.  We have also discussed the stochastic gradient descent algorithm and the mini-batch gradient descent as variations of batch gradient descent that can possibly reduce the time to convergence of the algorithm.

In this post we will summarize what we have discussed so far, and focus on the results that we have obtained from the various gradient descent algorithms.

All the code that we have written so far is available in the [GitHub repository](https://github.com/carmelgafa/ml_from_scratch/tree/master/algorithms/linear_regression).

## Data Generation

We have created two algorithms to generate the data for the linear regression problem, for a univariate and a two-feature case. Noise is applied to the data to make it more realistic. In the two-dimensional case, we are sampling a percentage of the data points as our training set.

| ![image](/post/img/ml_linearreg_gradientdescent_analysis_gen1.jpeg) |
|:--:|
| Generation of univariate training set from $y = 150 + 20x + \xi $|

| ![image](/post/img/ml_linearreg_gradientdescent_analysis_gen2.jpeg) |
|:--:|
| Generation of two-feature training set from $y = 12 + 5x_1 -3x_2 + \xi $|
|-|
The univariate case was generated because of its simplicity and also because it is possible to draw the cost function as a function of the coefficients $a_0$ and $a_1$.
