---
title: "Ml_linearreg_multivariatedescent"
date: 2022-01-12
tags: []
draft: true
---

We have discussed the multivariate linear regression problem in the [previous posts](/post/ml_linearreg_multivariate), and we have seen that in this case the hypothesis function becomes:

$$\hat{y} = a_0 + a_1 x_1 +  a_2 x_2 + \dots +  a_n x_n$$

we have seen that if, for convenience, we define $x_0 = 1, \forall i$, if we consider a training example,

$$x \in \textbf{R}^{n+1}=
\begin{pmatrix}
1 \\\\
x_1 \\\\
x_2 \\\\
\vdots \\\\
x_n
\end{pmatrix} = \textbf{X}$$

and 

$$
\beta =
\begin{pmatrix}
a_0 \\\\
a_1 \\\\
\vdots \\\\
a_n
\end{pmatrix} \in \textbf{R}^{n+1}
$$

$$\hat{y} = \beta^{T} \textbf{X}$$

It is very simple to generalize the gradient descent algorithm to the multivariate linear regression problem, thus obtaining the following:

repeat until convergence
  - $$ a_0 := a_0 - \frac{\alpha}{n} \sum_{i=1}^{n} ( \hat{y}_i-y_i)$$
  - $$ a_1 := a_1 - \frac{\alpha}{n} \sum_{i=1}^{n} ( \hat{y}_i-y_i)(x_i)$$
  - $$\dots$$
  - $$ a_n := a_n - \frac{\alpha}{n} \sum_{i=1}^{n} ( \hat{y}_i-y_i)(x_n)$$