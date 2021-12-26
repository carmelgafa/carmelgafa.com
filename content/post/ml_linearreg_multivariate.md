---
title: "Liner Regression, Part 4 - The Multi-variable scenario"
date: 2021-12-22
tags: [machine-learning, linear-regression, multi-variable]
draft: false
---

### Introduction

In previous posts we discussed the [univariate linear regression model](/post/ml_linearreg_univariatederivation) and how we can [implement the model in python](/post/ml_linearreg_univariatepython).

We have seen how we can fit a line, $ \hat{y} = a_0 + a_1 x$, to a dataset of given points, and how linear regression techniques estimate the values of $a_0$ and $a_1$ using the cost functions. We have seen that the residual is the difference between the observed values and the predicted values, that is, for any point $i$,

$$e_i = y_i - \hat{y_i}$$

We have looked at the Mean Square Error, the sum of the squared residuals divided by the number of points; hence our objective is to make the aggregation of residuals as small as possible.

$$argmin_{a_0, a_1} \frac{\sum_{i=1}^{n} (y_i-a_0-a_1 x_i)^{2} }{n}$$

we have seen that when we differentiate the cost function with respect to $a_0$ and $a_1$,

$$ a_0= \bar{y} - a_1 \bar{x}$$

and

$$ a_1 =\frac{ \sum_{i=1}^{n} x_i y_i - n\bar{x}\bar{y}}{\sum_{i=1}^{n} x_i^2 - n\bar{x}^2}$$

### Multi-variable Case

Most real-world problems have multiple features, and therefore our approximation is a hyperplane, which is a linear combination of the features, expressed as

$$\hat{y} = a_0 + a_1 x_1 +  a_2 x_2 + \dots +  a_n x_n$$

Hence if we define,
$$
\textbf{Y} =
\begin{pmatrix}
y_1 \\\\
y_2 \\\\
\vdots \\\\
y_n
\end{pmatrix}
$$

$$
\textbf{X} =
\begin{pmatrix}
1 & x_{11} & x_{12} & \dots & x_{1m} \\\\
1 & x_{21} & x_{22} & \dots & x_{2m} \\\\
\vdots & \vdots & \vdots & \vdots & \vdots \\\\
1 & x_{n1} & x_{n2} & \dots & x_{nm}
\end{pmatrix}
$$

$$
\beta =
\begin{pmatrix}
a_0 \\\\
a_1 \\\\
\vdots \\\\
a_n
\end{pmatrix}
$$

then,

$$\hat{\textbf{Y}} = \textbf{X} \beta $$

the residuals

$
\textbf{E} =
\begin{pmatrix}
e_1 \\\\
e_2 \\\\
\vdots \\\\
e_n
\end{pmatrix}
$=$
\begin{pmatrix}
y_1 - \hat{y}_1 \\\\
y_2 - \hat{y}_2 \\\\
\vdots \\\\
y_n - \hat{y}_n
\end{pmatrix}
$= $\textbf{Y}-\hat{\textbf{Y}}$

We will here introduce the residual sum-of-squares cost function, which is very similar to the mean square error cost function, but it is defined as

$$RSS = \sum_{i=1}^{n} e_i^2$$

We have noticed in the previous cases that the effect of considering the mean is eliminated during the derivation of the cost function and equating to zero.

we also notice that

$$
RSS = \textbf{E}^T \textbf{E}\\\\
 = (\textbf{Y}-\hat{\textbf{Y}})^T(\textbf{Y}-\hat{\textbf{Y}})\\\\
 = (\textbf{Y}- \textbf{X} \beta )^T (\textbf{Y}- \textbf{X} \beta )\\\\
 = \textbf{Y}^T\textbf{Y}-\textbf{Y}^T\textbf{X} \beta^T - \textbf{X}^T \textbf{Y} + \beta^T\textbf{X}^T\textbf{X} \beta
 $$

Matrix Differentiation

Before we continue, we will first remind ourselves of the following:

If we are given two independent matrices $x$, and $A$, where $x$ is an m by 1 matrix and $A$ is an n by m matrix, then;

for $y=A$ $\rightarrow$ $\frac{dy}{dx}=0$,

for $y=Ax$ $\rightarrow$ $\frac{dy}{dx}=A$,

for $y=xA$ $\rightarrow$ $\frac{dy}{dx}=A^T$,

for $y=x^TAx$ $\rightarrow$ $\frac{dy}{dx}=2x^TA$,

Hence, differentiating the cost function with respect to $\beta$,

$$
\frac{\partial RSS}{\partial\beta} = 0 -\textbf{Y}^T\textbf{X} - (\textbf{X}^T \textbf{Y})^T +  2 \beta^T\textbf{X}^T\textbf{X}\\\\
= -\textbf{Y}^T\textbf{X} - \textbf{Y}^T \textbf{X} +  2 \beta^T\textbf{X}^T\textbf{X}\\\\
= - 2 \textbf{Y}^T \textbf{X} +  2 \beta^T\textbf{X}^T\textbf{X}
$$

for minimum $RSS$, $ \frac{\partial RSS}{\partial\beta} = 0$, hence

$$
2 \beta^T\textbf{X}^T\textbf{X} = 2 \textbf{Y}^T \textbf{X}\\\\
\beta^T\textbf{X}^T\textbf{X} = \textbf{Y}^T \textbf{X}\\\\
\beta^T = \textbf{Y}^T \textbf{X}(\textbf{X}^T\textbf{X})^{-1}\\\\
$$
and therefore

$$
\beta = (\textbf{X}^T\textbf{X})^{-1} \textbf{X}^T \textbf{Y}\\\\
$$

### Two-variable case equations

For the scenario where we have only 2 features, so that $\hat{y} = a_0 + a_1 x_1 + a_2 x_2$, we can obtain the following equations for the parameters $a_0$, $a_1$ and $a_2$:

$$a_1 = \frac{ \sum_{i=1}^{n} x_{2i}^2  \sum_{i=1}^{n} x_{1i}y_i - \sum_{i=1}^{n} x_{1i}x_{2i} \sum_{i=1}^{n} x_{2i}y_i }
{\sum_{i=1}^{n}x_{1i}^2 \sum_{i=1}^{n}x_{2i}^2 - \sum_{i=1}^{n} (x_{1i}x_{2i})^2}$$

$$a_2 = \frac{ \sum_{i=1}^{n} x_{1i}^2  \sum_{i=1}^{n} x_{2i}y_i - \sum_{i=1}^{n} x_{1i}x_{2i} \sum_{i=1}^{n} x_{1i}y_i }
{\sum_{i=1}^{n}x_{1i}^2 \sum_{i=1}^{n}x_{2i}^2 - \sum_{i=1}^{n} (x_{1i}x_{2i})^2}$$

and

$$ a_0 = \bar{\textbf{Y}} - a_1 \bar{\textbf{X}}_1 - a_2 \bar{\textbf{X}}_2$$

It is evident that finding the parameters becomes more difficult as we add more features.
