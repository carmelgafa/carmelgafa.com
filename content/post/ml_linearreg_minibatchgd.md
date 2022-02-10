---
title: "Ml_linearreg_minibatchgd"
date: 2022-02-09
tags: []
draft: true
---

In the [last post](/post/ml_linearreg_stochasticgd.md) we compared the stochastic gradient descent algorithm to the batch gradient descent algorithm that we has discussed in [a previous post](/post/ml_linearreg_gradientdescent.md). We discussed that as the size of the training dataset increases, batch gradient descent, where we use all the examples of the training set in each iteration, becomes very computationally expensive and that we can therefore use stochastic gradient descent, where we use one example of the training set in each iteration, to have a more efficient way to approach the coefficients of our hypothesis function.

In this post we will discuss mini-batch gradient descent, where we use a number $k$ of the training set examples in each iteration, which is a variation of the thoughts of stochastic gradient descent. We will discuss the general idea of mini-batch gradient descent and how to implement it in Python.

Therefore we can consider our training dataset as a collection of $b/m$ batches;

$$\textbf{X} = \begin{pmatrix}
x_0^{(1)} &\dots & x_0^{(k)} & x_0^{(k+1)} & \dots & x_0^{(2k)} & \dots \dots & x_0^{(m)}\\\\
x_1^{(1)} &\dots & x_1^{(k)} & x_1^{(k+1)} & \dots & x_1^{(2k)} & \dots \dots & x_1^{(m)}\\\\
\dots     &\dots & \dots     & \dots       & \dots & \dots      & \dots \dots & \dots    \\\\
x_n^{(1)} &\dots & x_n^{(k)} & x_n^{(k+1)} & \dots & x_n^{(2k)} & \dots \dots & x_n^{(m)}\\\\
\end{pmatrix}$$
$$Dim:[n \times m]$$

Calculating the hypothesis function for a mini-batch of training data, we can write the following equation:

We can now calculate the hypothesis function as a matrix multiplication:

$$\begin{pmatrix}
\hat{y}^{(1)} \\\\
\hat{y}^{(2)} \\\\
\vdots \\\\
\hat{y}^{(k)}
\end{pmatrix} =
\begin{pmatrix}
x_0^{(1)} & x_1^{(1)} & \dots & x_n^{(1)} \\\\
x_0^{(2)} & x_1^{(2)} & \dots & x_n^{(2)} \\\\
\vdots & \vdots & \vdots & \vdots \\\\
x_0^{(k)} & x_1^{(k)} & \dots & x_n^{(k)}
\end{pmatrix}
\begin{pmatrix}
a_0 \\\\
a_1 \\\\
\vdots \\\\
a_n
\end{pmatrix}
$$
$$Dim:[k \times n] \cdot [n \times 1]$$

