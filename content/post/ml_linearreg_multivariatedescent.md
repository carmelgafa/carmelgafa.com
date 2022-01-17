---
title: "Ml_linearreg_multivariatedescent"
date: 2022-01-12
tags: []
draft: true
---

We have discussed the multivariate linear regression problem in the [previous posts](/post/ml_linearreg_multivariate), and we have seen that in this case the hypothesis function becomes:

$$\hat{y} = a_0 + a_1 x_1 +  a_2 x_2 + \dots +  a_n x_n$$

If we define $x_0$, such that $x_0 = 1$, then the hypothesis function becomes:

$$\hat{y} = a_0 x_0 + a_1 x_1 +  a_2 x_2 + \dots +  a_n x_n$$


Let us now consider a dataset of $m$ points. we can therefore calculate the hypothesis function for each point,

$$\hat{y^1} = a_0 x_0^1 + a_1 x_1^1 +  a_2 x_2^1 + \dots +  a_n x_n^1$$

$$\hat{y^2} = a_0 x_0^2 + a_1 x_1^2 +  a_2 x_2^2 + \dots +  a_n x_n^2$$

$$\dots$$

$$\hat{y^m} = a_0 x_0^m + a_1 x_1^m +  a_2 x_2^m + \dots +  a_n x_n^m$$

where 
$x_1^i$ is the first feature of the $i$th point and,

$x_0^i = 1 \forall i$.

we can express the equation of the hypothesis function shown above as a matrix multiplication:


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