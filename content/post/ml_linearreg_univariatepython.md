---
title: "Linear Regression - Univariate Solution Implementation in Python"
date: 2021-12-21
tags: [machine-learning, linear-regression, python]
draft: true
---

This post continues from the derivation of the univariate linear regression model as explained in the [previous post](/post/ml_linearreg_univariatederivation). Here we will use the equations derived and the in practice to implement the model.

### Univarite Function

We start this discussion by considering the function used in this post. The function that we will use is

$$y = 2x + 15 + \xi$$

Where $\xi$ is a random variable that will introduce noise to the data. The data is generated using the following code.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


x = np.arange(0, 10.1, 0.1)

# generate a noisy line
y = (2*x+ 15)+ np.random.randn(len(x))

y_line = 2*x + 15

plt.plot(x, y)
plt.plot(x, y_line, '--')


plt.xlim([0, 11])
plt.ylim([0, 40])

plt.show()

pd.DataFrame(y).to_csv("data.csv", header=False, index=True)
```

Note that the data is saved in a CSV file to use subsequently.

### Finding the Linear Regression Coefficients

In the [previous post](/post/ml_linearreg_univariatederivation) we discussed how if we are given a set of points;

$$ (x_1, y_1), (x_2, y_2), \dots, (x_i, y_i), \dots ,(x_n, y_n)$$

It is possible to fit the line 

$$ \hat{y} = a_0 + a_1 x$$ 

through the data such that:

$$a_0 = \frac{\sum_{i=1}^{n} x_i \sum_{i=1}^{n} y_i - \sum_{i=1}^{n} x_i \sum_{i=1}^{n} y_i x_i}{n \sum_{i=1}^{n}  x_i^2 - (\sum_{i=1}^{n} x_i)^2} $$

and

$$a_1 = \frac{n \sum_{i=1}^{n} x_i - \sum_{i=1}^{n} x_i \sum_{i=1}^{n} y_i}{n \sum_{i=1}^{n}  x_i^2 - (\sum_{i=1}^{n} x_i)^2} $$

We notice that the equations are made of several summations, and it might be helpful to list the most important ones out.

|    |
|:---:|
|$$\sum_{i=1}^{n} x_i$$|
|$$\sum_{i=1}^{n} y_i$$|
|$$\sum_{i=1}^{n} y_i x_i$$|
|$$\sum_{i=1}^{n}  x_i^2$$|

We notice that terms such as $n \sum_{i=1}^{n}  x_i^2$ or $\sum_{i=1}^{n} x_i^2$ can be generated from the functions listed above.
