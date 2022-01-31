---
title: "Linear Regression Part 8 - Stochastic Gradient Descent"
date: 2022-01-31T16:24:09+01:00
tags: [machine-learning, linear-regression, python]
draft: true
---

In a previous post we have discussed the gradient descent algorithm for linear regression applied to multiple features. As the size of the training dataset increases, gradient descent becomes very computationally expensive. In particular, if we consider the computation of the cost function,

$$J = \frac{1}{2m} \sum_{i=1}^m \left( \hat{y}^{(i)} - {y}^{(i)} \right)^2$$,

which is a calculation that we need to compute for each training data iteration, we can can see that the cost function is expensive as m increases.

Stochastic gradient descent is a variant of gradient descent where coefficients are updated after each observation, and it is therefore better suited for large datasets. Therefore if we consider an observation $i$ of the training dataset, we can calculate the cost of the observation, or how well our hypothesis function predicts the observation as,

$$Cost(\beta, (x^{(i)}, y{(i)})) = \frac{1}{2} \left(\hat{y}^{(i)} - {y}^{(i)} \right)^2$$

We can therefore calculate the cost function as

$$J(\beta) = \frac{1}{m} \sum_{i=1}^m \left( \hat{y}^{(i)} - {y}^{(i)} \right)^2$$

Random Gradient Descent can be implemented as follows:

- Shuffle the training dataset, thus making sure that the order of the observations is random.

- For each observation:
  - $$\begin{pmatrix}
a_0 \\\\
a_1 \\\\
\vdots \\\\
a_n
\end{pmatrix} :=
\begin{pmatrix}
a_0 \\\\
a_1 \\\\
\vdots \\\\
a_n
\end{pmatrix} -
\alpha
\begin{pmatrix}
\hat{y}^{(i)} - {y}^{(i)}
\end{pmatrix}
\cdot
\begin{pmatrix}
x_0^{(i)}\\\\
x_1^{(i)}\\\\
\vdots\\\\
x_n^{(i)}
\end{pmatrix}
$$

where $m$ is the number of training data points.
