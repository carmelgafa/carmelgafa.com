---
title: "Linear Regression, Part 3 - Univariate Solution Implementation in Python"
date: 2021-12-21
tags: [machine-learning, linear-regression, python]
draft: false
---

This post continues from the derivation of the univariate linear regression model as explained in the [previous post](/post/ml_linearreg_univariatederivation). Here we will use the equations derived and the in practice to implement the model.

### Univarite Function

We start this discussion by considering the function used in this post. The function that we will use is

$$y = 2x + 15 + \xi$$

Where $\xi$ is a random variable that will introduce noise to the data. The data is generated using the following code.

```Python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# x between 0 and 100 in steps of 1
x = np.arange(0, 101, 1)

# generate a noisy line
l = (2*x) + 15

random_multiplier = 5
e = np.random.randn(len(x))*random_multiplier
y = l + e

# plot the data
plt.plot(x, y)
plt.plot(x, l, '--')
plt.xlim([0, 101])
plt.ylim([0, 200])
plt.show()

# save the data to a csv file
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
|$$\sum_{i=1}^{n} x_i^2$$|

We notice that terms such as $n \sum_{i=1}^{n}  x_i^2$ or $\sum_{i=1}^{n} x_i^2$ can be generated from the functions listed above.

We can therefore implement the equations above in Python as follows. The relevant information is stored in a pandas dataframe to make it easier to access.

```python
import pandas as pd

# import data from csv
data = pd.read_csv("data.csv")
data.columns=['x', 'y']

# add new columns required to solve the problem
data['x_sq'] = data['x']**2
data['xy'] = data['x']*data['y']


# calculate the sums of the data
sum_x = data['x'].sum()
sum_y = data['y'].sum()
sum_x_sq = data['x_sq'].sum()
sum_xy = data['xy'].sum()

n = len(data)
print(f'sum_x: {sum_x}, sum_y: {sum_y}, sum_x_sq: {sum_x_sq}, sum_xy: {sum_xy}, n: {n}')

# calculate the slope and intercept
a_0 = (sum_x_sq*sum_y - sum_x*sum_xy)/(n*sum_x_sq - sum_x**2)

a_1 = (n*sum_xy - sum_x*sum_y)/(n*sum_x_sq - sum_x**2)

print(f'a_0: {a_0}, a_1: {a_1}')
```

The coefficients obtained depend on the noise available in the data. With the code above, we can see that the coefficients are around $a_0 = 15$ and $a_1 = 2$.
