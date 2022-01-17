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

we can express the equation of the hypothesis function shown above as a matrix multiplication. Let us start by defining the matrix $X$ as the matrix of features, such that 


$$\textbf{X} = \begin{pmatrix}
x_0^1 & x_0^2 & \dots & x_0^m \\\\
x_1^1 & x_1^2 & \dots & x_1^m \\\\
\vdots & \vdots & \vdots & \vdots \\\\
x_n^1 & x_n^2 & \dots & x_n^m
\end{pmatrix}$$

where, for example, $x^2$ is the second feature point,

$$x^2 = \begin{pmatrix}
x_0^2 \\\\
x_1^2 \\\\
\vdots \\\\
x_n^2
\end{pmatrix}$$


the matrix of coefficients $\beta$ as:

$$\beta = \begin{pmatrix}
a_0 \\\\
a_1 \\\\
\vdots \\\\
a_n
\end{pmatrix}$$

and the matrix of predictions $\hat{y}$ as:

$$\hat{y} = \begin{pmatrix}
\hat{y}^1 \\\\
\hat{y}^2 \\\\
\vdots \\\\
\hat{y}^m
\end{pmatrix}$$



We can now calculate the hypothesis function as a matrix multiplication:

$$\begin{pmatrix}
\hat{y}^1 \\\\
\hat{y}^2 \\\\
\vdots \\\\
\hat{y}^m
\end{pmatrix} = 
\begin{pmatrix}
a_0 & a_1 & \dots & a_n
\end{pmatrix} 
\begin{pmatrix}
x_0^1 & x_0^2 & \dots & x_0^m \\\\
x_1^1 & x_1^2 & \dots & x_1^m \\\\
\vdots & \vdots & \vdots & \vdots \\\\
x_n^1 & x_n^2 & \dots & x_n^m
\end{pmatrix}$$

or

$$\beta^T \cdot X = \hat{Y}$$


We can now define the matrix of residuals as:

$$\textbf{R} = \begin{pmatrix}
\hat{y}^1 - {y}^1 \\\\
\hat{y}^2 - {y}^2 \\\\
\vdots \\\\
\hat{y}^m - {y}^m
\end{pmatrix}$$

and if we define a matrix of $\delta$
