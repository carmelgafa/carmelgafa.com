---
title: Interval Type-2 Fuzzy Sets Type Reduction
description: Jerry Mendel's IT2FS type reduction examples
date: "2020-02-03T16:17:59+01:00"
tags: [type2-fuzzy-library, type2-fuzzy, fuzzy, python, IT2FS]
---
Jerry Mendel's [book](https://www.amazon.co.uk/Uncertain-Rule-Based-Fuzzy-Systems-Introduction/dp/3319513699/ref=sr_1_3?keywords=jerry+mendel&qid=1580743185&sr=8-3) can be safely considered as the Type-2 Fuzzy Logic bible. It is not an easy or inexpensive read, but definitely the best way to get to know the world of Type-2 Sets.

In this post a book example, where the centroid of a number of Interval Type-2 sets was calculated is replicated using the Type-2 Library.

Type reduction in these cases is carried out by using the function ```it2_kernikmendel_reduce(interval_set)```, and a crisp set, the centroid, is returned back.


```python

'''
This module executes the karnik-mendel type reduction 
algorithm and compares the results with those obtained 
in mendel's book

References:
-----------
Mendel, Jerry M. Uncertain rule-based fuzzy systems. 
Springer, Cham, 2017. pages 261-262
'''

import numpy as np
from numpy.random import normal
import math
import matplotlib.pyplot as plt
from type2fuzzy import IntervalType2FuzzySet
from type2fuzzy import it2_kernikmendel_reduce

def gaussian(x, mean, sigma):
  g = np.exp(-0.5*(((x - mean)/sigma)**2))
  return g

def generate_sets_uncertain_mean(m1_m2_list):
  
  it2fs_list = []
  
  x= np.linspace(0,10,101)

  for m1_m2 in m1_m2_list:
    m1 = m1_m2[0]
    m2 = m1_m2[1]

    g1 = gaussian(x, m1 ,1)
    g2 = gaussian(x, m2 ,1)
  
    hmf = np.maximum(g1,g2)
  
    one_indexes = np.where(hmf==1)[0]

    if len(one_indexes) > 1:
      hmf[one_indexes[0]: one_indexes[1]] = 1
  
    lmf = np.minimum(g1,g2)

    it2fs_list.append(
      IntervalType2FuzzySet.from_hmf_lmf(x, hmf, lmf))

  return it2fs_list

def generate_sets_uncertain_variance(m1_m2_list):
  
  it2fs_list = []
  
  x= np.linspace(0,10,101)


  for s1_s2 in m1_m2_list:
    s1 = s1_s2[0]
    s2 = s1_s2[1]

    g1 = gaussian(x, 5 ,s1)
    g2 = gaussian(x, 5 ,s2)
  
    hmf = np.maximum(g1,g2)
    lmf = np.minimum(g1,g2)

    it2fs_list.append(
      IntervalType2FuzzySet.from_hmf_lmf(x, hmf, lmf))

  return it2fs_list

def it2fs_centroid(it2fs_list):
  
  for it2fs in it2fs_list:
    centroid = it2_kernikmendel_reduce(
      it2fs, information='none', precision=4)
        print(f'Centroid: {centroid}')

# test 1 - table 9-1
# results obtained:
# [5.00000]
# [4.87498, 5.12502]
# [4.74952, 5.25048]
# [4.62265, 5.37735]
# [4.49285, 5.50715]
# [4.21675, 5.78325]
# [3.90697, 6.09303]
# [3.55194, 6.44806]
# [3.15053, 6.84947]
m1_m2_list = [(5,5), (4.875,5.125), (4.75, 5.25), 
  (4.625, 5.375), (4.5, 5.5), (4.25, 5.75), (4,6), 
  (3.75, 6.25), (3.5 ,6.5)]
it2fs_list_m = generate_sets_uncertain_mean(m1_m2_list)
print('Uncertain Mean Results')
it2fs_centroid(it2fs_list_m)
print('\n')

# test 2 - table 9-2
# results obtained:
# [5.00000]
# [4.80054, 5.19946]
# [4.60079, 5.39921]
# [4.39849, 5.60151]
# [4.18488, 5.81512]
# [3.93441, 6.06559]
# [3.59388, 6.40612]
s1_s2_list = [(1,1), (0.875,1.125), (0.75, 1.25), 
  (0.625,1.375), (0.5, 1.5), (0.375,1.625), (0.25,1.75)]
it2fs_list_v = generate_sets_uncertain_variance(s1_s2_list)
print('Uncertain Variance Results')
it2fs_centroid(it2fs_list_v)

```
