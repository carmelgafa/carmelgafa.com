---
title: "Linear Regression, Part 6 - Ml_linearreg_gradientdescent"
date: 2022-01-06T11:08:45+01:00
tags: [machine-learning, linear-regression, python]
draft: true
---

We started this series of posts with an examination of [Cost Functions](/post/ml_linearreg_costfunctions), and then moved on to [derive](/post/ml_linearreg_univariatederivation) and [implement](/post/ml_linearreg_univariatepython) the solution to the linear regression problem for a single variable We extended this to a multi-variable linear regression problem, and we [derived](/post/ml_linearreg_multivariate) and [implemented](/post/ml_linearreg_multivariatepython) the solution for this case. Our final comment was that as the number of variables increases, the solution becomes computationally prohibitive.

In this post we will look at **gradient descent**, and iterative optimization algorithm that is used to find the minimum of a function. The basic idea is to iteratively move in the opposite direction of the gradient at a given point, thus moving in the direction of the steepest descent.

### Recap of the univariate linear regression problem

We have seen in the previous posts that given a hypothesis function 

$$ a_0 + a_1 x $$

we can vary our parameters $a_0$ and $a_1$ minimize a cost function. The cost function that we used in the previous posts is the Mean Square Error,

$$MSE = \frac{\sum_{i=1}^{n} ( \hat{y}_i-y_i )^{2} }{n}$$

In this post, we will use a slightly different version of MSE, that is,

$$MSE = \frac{\sum_{i=1}^{n} ( \hat{y}_i-y_i )^{2} }{2n}$$

The purpose of factor of two in the denominator is purely to simplify the calculations that we will meet further on. Adding a factor of two to the denominator does not modify the effect of the cost function, but as the MSE contains a squared term, the derivation will produce a factor of two in the numerator. Having a scaled MSE will mean that the factor of 2 will cancel out when deriving.

### Gradient Descent

The basic algorithm for gradient descent is simple, and we will use the following notation:

- start initial values for the parameters $a_0$ and $a_1$
- keep changing the parameters until the cost function is minimized

We can formally write the algorithm as follows:

repeat until convergence
  - $$ a_0 := a_0 - \alpha \frac{\partial}{\partial a_0} MSE(a_0, a_1)$$
  - $$ a_1 := a_1 - \alpha \frac{\partial}{\partial a_1} MSE(a_0, a_1)$$

#### A note about the value of $\alpha$

The value of $\alpha$ is the step size. It affects the precision and the speed of convergence, and should be tuned to the problem at hand.

- The smaller the step size, the more accurate the solution will be, but the longer it will take to converge.
- The larger the step size, gradient descent can overshoot and may fail to converge or it might also diverge.

One important thing to note that the term 

  $$  \alpha \frac{\partial}{\partial a} MSE(a_0, a_1)$$

will become smaller in magnitude as the result moves closer to a minimum. Hence, gradient descent will take smaller stems as a local minimum is approached, and therefore there is no need to adjust the step size by adjusting $\alpha$.