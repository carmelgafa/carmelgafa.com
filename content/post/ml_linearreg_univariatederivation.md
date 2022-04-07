---
title: Liner Regression, Part 2 - Deriving the Univariate case
date: 2021-12-20
tags: [machine-learning, linear-regression, gradient-descent, python]
draft: false
---

This post is a continuation of [a previous post](/post/ml_linearreg_costfunctions) where the cost functions used in linear regression scenarios are used. We will start by revisiting the mean square error (MSE) cost function;

$$MSE = \frac{\sum_{i=1}^{n} ( \hat{y}_i-y_i )^{2} }{n}$$

which, as explained in the previous post, is

$$MSE = \frac{\sum_{i=1}^{n} (y_i-a_0-a_1 x_i)^{2} }{n}$$

The objective is to adjust $a_0$ and $a_1$ such that the MSE is minimized. This is achieved by deriving the MSE with respect to $a_0$ and $a_1$, and finding the minimum case by equating to zero.

$$\frac{\partial MSE}{\partial a_0} = 0$$

and

$$\frac{\partial MSE}{\partial a_1} = 0$$

Now,

$$\frac{\partial MSE}{\partial a_0} = \frac{\sum_{i=1}^{n} 2( y_i-a_0-a_1 x_i )(-1) }{n}$$

$$ = \frac{2}{n} \sum_{i=1}^{n} -y_i+a_0+a_1 x_i  $$

At minimum, $\frac{\partial MSE}{\partial a_0} = 0$, i.e.

$$\frac{2}{n} \sum_{i=1}^{n} -y_i+a_0+a_1 x_i = 0 $$

$$\sum_{i=1}^{n} - y_i+a_0+a_1 x_i = 0 $$

$$-\sum_{i=1}^{n} y_i + \sum_{i=1}^{n} a_0 + \sum_{i=1}^{n}  a_1 x_i = 0 $$

$$\sum_{i=1}^{n} a_0 + \sum_{i=1}^{n}  a_1 x_i = \sum_{i=1}^{n} y_i$$

or

$$ n a_0 + a_1 \sum_{i=1}^{n} x_i = \sum_{i=1}^{n} y_i$$

Similarly,

$$\frac{\partial MSE}{\partial a_1} = \frac{\sum_{i=1}^{n} 2( y_i-a_0-a_1 x_i )(-x_i) }{n}$$

$$ = \frac{2}{n} \sum_{i=1}^{n} ( y_i-a_0-a_1 x_i )(-x_i)$$

$$ = \frac{2}{n} \sum_{i=1}^{n} -x_i y_i + a_0 x_i + a_1 x_i^2 $$

At minimum, $\frac{\partial MSE}{\partial a_1} = 0$, i.e.

$$\frac{2}{n} \sum_{i=1}^{n} -x_i y_i + a_0 x_i + a_1 x_i^2 = 0 $$

$$\sum_{i=1}^{n} -x_i y_i + a_0 x_i + a_1 x_i^2 = 0 $$

$$\sum_{i=1}^{n} -x_i y_i + a_0 x_i + a_1 x_i^2 = 0 $$

$$ - \sum_{i=1}^{n} x_i y_i + \sum_{i=1}^{n} a_0 x_i + \sum_{i=1}^{n} a_1 x_i^2 = 0 $$

$$\sum_{i=1}^{n} a_0 x_i + \sum_{i=1}^{n} a_1 x_i^2 = \sum_{i=1}^{n} x_i y_i $$

This can be written in matrix form as

$
\begin{pmatrix}
n & \sum_{i=1}^{n} x_i \\\\
\sum_{i=1}^{n} x_i & \sum_{i=1}^{n} x_i^2
\end{pmatrix}
$
$
\begin{pmatrix}
a_0 \\\\
a_1
\end{pmatrix} =
$
$
\begin{pmatrix}
\sum_{i=1}^{n} y_i \\\\
\sum_{i=1}^{n} x_i y_1
\end{pmatrix}
$

This can be solved using Cramer's rule.
$$
a_0 = \frac
{
\begin{vmatrix}
\sum_{i=1}^{n} y_i & \sum_{i=1}^{n} x_i\\\\
\sum_{i=1}^{n} y_i x_i & \sum_{i=1}^{n} x_i^2
\end{vmatrix}
}{\sum_{i=1}^{n} n x_i^2 - (\sum_{i=1}^{n} x_i)^2}
$$

$$=\frac{\sum_{i=1}^{n} x_i \sum_{i=1}^{n} y_i - \sum_{i=1}^{n} x_i \sum_{i=1}^{n} y_i x_i}{\sum_{i=1}^{n} n x_i^2 - (\sum_{i=1}^{n} x_i)^2} $$

Similarly,

$$
a_1 = \frac
{
\begin{vmatrix}
n & \sum_{i=1}^{n} y_i\\\\
\sum_{i=1}^{n} x_i & \sum_{i=1}^{n} x_i y_i 
\end{vmatrix}
}{\sum_{i=1}^{n} n x_i^2 - (\sum_{i=1}^{n} x_i)^2}
$$

$$=\frac{n \sum_{i=1}^{n} x_i y_i - \sum_{i=1}^{n} x_i \sum_{i=1}^{n} y_i}{\sum_{i=1}^{n} n x_i^2 - (\sum_{i=1}^{n} x_i)^2} $$

$$ =\frac{ \sum_{i=1}^{n} x_i y_i - n\bar{x}\bar{y}}{\sum_{i=1}^{n} x_i^2 - n\bar{x}^2}$$

We also note that as,

$$ n a_0 + a_1 \sum_{i=1}^{n} x_i = \sum_{i=1}^{n} y_i$$

$$ n a_0 = \sum_{i=1}^{n} y_i - a_1 \sum_{i=1}^{n} x_i$$

$$  a_0 = \frac{\sum_{i=1}^{n} y_i}{n} - a_1 \frac{\sum_{i=1}^{n} x_i}{n}$$

$$ = \bar{y} - a_1 \bar{x}$$
