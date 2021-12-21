---
title: "Type Reduction of Interval Type-2 Fuzzy Sets"
date: 2021-12-21
tags: []
draft: true
---

In this post we will look aw Interval Type-2 fuzzy set (IT2FS) and its reduction. We have already discussed the basis of Type-2 fuzzy sets in a [previous post](post/type2fuzzy_set.md) and we have seen that a general type-2 fuzzy set can be deined as follows:

$$\tilde{A}=\int_{x\in X}\int_{u\in J_{x}} \mu_{\tilde{A}}(x,u) / (x,u)$$

where $J_{x}\subseteq[0,1]$

And if, as an example we consider the following general type-2 fuzzy set:

```{math}
(1.0/0 + 0.5/0.2 + 0.3/0.4 + 0.1/0.6                       )/1+
(0.5/0 + 1.0/0.2 + 0.5/0.4 + 0.3/0.6 + 0.1/0.8             )/2+
(0.1/0 + 0.3/0.2 + 0.5/0.4 + 1.0/0.6 + 0.5/0.8 + 0.3/1.0   )/3+
(        0.1/0.2 + 0.3/0.4 + 0.5/0.6 + 1.0/0.8 + 0.5/1.0   )/4+
(1.0/0 + 0.5/0.2 + 0.3/0.4 + 0.1/0.6                       )/5
```

The set can be created and displayed using the type2fuzzy library as follows:

```python
from type2fuzzy import GeneralType2FuzzySet
from type2fuzzy.display.generaltype2fuzzysetplot import GeneralType2FuzzySetPlot
import matplotlib.pyplot as plt

# Create a general type-2 fuzzy set
gt2fs_rep =   '''
 (1.0/0 + 0.5/0.2 + 0.3/0.4 + 0.1/0.6                       )/1
+(0.5/0 + 1.0/0.2 + 0.5/0.4 + 0.3/0.6 + 0.1/0.8             )/2
+(0.1/0 + 0.3/0.2 + 0.5/0.4 + 1.0/0.6 + 0.5/0.8 + 0.3/1.0   )/3
+(        0.1/0.2 + 0.3/0.4 + 0.5/0.6 + 1.0/0.8 + 0.5/1.0   )/4
+(1.0/0 + 0.5/0.2 + 0.3/0.4 + 0.1/0.6                       )/5'''

gt2fs = GeneralType2FuzzySet.from_representation(gt2fs_rep)

# Plot the general type-2 fuzzy set
print(f'\nSet representation: {gt2fs}')

fig = plt.figure()
ax=fig.add_subplot(1,1,1)

set_plt = GeneralType2FuzzySetPlot(gt2fs)
set_plt.plot(ax)

plt.show()
```

which will result in the following plot:
![type-2 fuzzy set](/post/img/type2fuzzy_it2fs_typereduction_type2_set.png)

### Interval Type-2 fuzzy set (IT2FS)

An Interval Type-2 Fuzzy set (IT2FS) is a Type-2 fuzzy set where all the secondary grades are equal to 1. Secondary grades have therefore no information hence it is completely described by its Footprint of uncertainty and thus by its Lower and upper membership functions. Hence an IT2FS is defined as follows:

$$C_{\tilde{A}} = \int_{\theta_{1}\in J_{x_{1}}} \dots  \int_{\theta_{N}\in J_{x_{N}}} 1 / \frac{\sum_{i=1}^{N} x_i \theta_{i}}{\sum_{i=1}^{N} \theta_{i}}$$

we can see by inspection that the IT2FS equivalent of the general tye-2 fuzzy set under examiniation is the following:

```math
[0.00000, 0.60000]/1.0+
[0.00000, 0.80000]/2.0+
[0.00000, 1.00000]/3.0+
[0.20000, 1.00000]/4.0+
[0.00000, 0.60000]/5.0
```

This can be obtained by using the type2fuzzy library as follows:

```python
from type2fuzzy import IntervalType2FuzzySet
from type2fuzzy.display.intervaltype2fuzzysetplot import IntervalType2FuzzySetPlot

it2fs = IntervalType2FuzzySet.from_general_type2_set(gt2fs)

print(f'\nSet representation: {it2fs}')

fig = plt.figure()
ax=fig.add_subplot(1,1,1)

set_plt = IntervalType2FuzzySetPlot(it2fs)
set_plt.plot(ax)

plt.show()
```

which will plot the following:

![type-2 fuzzy set](/post/img/type2fuzzy_it2fs_typereduction_type2_it2fs.png)

### Centroid of an Interval Type-2 fuzzy set

An iterative procedure to find the centroid of an Interval Type-2 fuzzy is discussed in 

_"Karnik, N. N., & Mendel, J. M. (1998, May). Introduction to type-2 fuzzy logic systems. In 1998 IEEE international conference on fuzzy systems proceedings. IEEE world congress on computational intelligence (Cat. No. 98CH36228) (Vol. 2, pp. 915-920). IEEE."_

