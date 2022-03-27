---
title: "Paper Implementation - C. Wagner and H. Hagras. 'Toward general type-2 fuzzy logic systems based on zSlices.'"
date: 2022-03-27
tags: [type2-fuzzy, paper-workout, type2-fuzzy-library, fuzzy, python]
draft: false
---

The [paper](https://d1wqtxts1xzle7.cloudfront.net/31004495/wagner2010-with-cover-page-v2.pdf?Expires=1648384380&Signature=gfElNw7ckIhZsARlzuO3axX8j2sNdhhEXEXNxK-yPuUYbOuDHCvo4f6tCnViiUa6Di3kD1Gaz~kojOGHtr83n7FdwOO3OKcTDd918fwq~63EIdeBKywNWOjfaklzjfk2Swc5c9Quq6NKDgH5osOF~0aBTWbe~8Nd3RGaMu8tJvjtnBnnE4h8seUpZS-ZpEx6zaR6RUMjPPiNc2ZEKZI5podOlVrYsNkJNa4jL4A3LutyJoKiJnUbuJoF8VGJAYrWaN9G1ijIMZTxOUTpOCi99auZUIRyG3~gP~nJhglPVIX5kXnQqa-r2oYg72h4iR7PBh2jmpU463xjY4pNrzlpvw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) by Wagner and Hagras explains the concept of zSlices as a way to implement type-2 fuzzy logic systems based on general type-2 fuzzy sets. The paper's appendices contain numeric examples of centroid computation based on zSlices. This post describes the steps involved in implementing the paper by Wagner and Hagras. It also lists the results obtained and compares them with those listed in the original document.

### Centroid of a zSliced Based Type-2 Set

The general type-2 fuzzy set used in these examples will result in the potential computation of the centroid of $3^{11}$ wavy slices. For this reason, a crudely discretized version of the set that contains only embedded type-2 fuzzy sets was used.

The working in the paper starts with the classical wavy-slice equation to type-reduce a general type-2 fuzzy set,

$$C_{\tilde{G}} = \int_{u_1\in J_{x_1}} \dots \int_{u_n\in J_{x_n}}  f_{x_1}(u_{1}) \star \dots \star f_{x_n}(u_{n}) / \frac{\sum_{i=1}^{N} u_{i}x_{i}}{\sum_{i=1}^{N} u_{i}}$$

and used this method to find the centroid of a general type-2 fuzzy set,

$$\tilde{G}_D = (0.33/0.0)/1 + (0.66/0)/2 + (0.66/0+0.66/0.5)/3 + (0.33/0 + 0.66/0.5)/4 +$$
$$(1/1)/5 + (0.33/0.5)/6 + (0.33/0 + 1/0.5)/7 + (0.66/0)/8 + (0.66/0)/9$$

we obtained the following type-reduced result:

$$C_{\tilde{G}_D} = 0.33/5.33 + 0.33/5 + 0.33/4.75 + 0.33/4.6 + 0.33/5.75 + 0.33/5.4 + 0.33/5.2 + 0.33/5$$

Which yields a defuzzified crisp value of $O{C_{\tilde{G}_D}} = 5.12875$ after using a standard centroid defuzzifier.

Here, however, we notice a slight mistake in this calculation. The type-reduced centroid, $C_{\tilde{G}_D}$ contains two values at $x=5$ both of 0.33.
However, in the original paper by Karnik, Mendel, and Qilian (Karnik, Nilesh Naval, Jerry M. Mendel, and Qilian Liang. "Type-2 fuzzy logic systems." IEEE transactions on Fuzzy Systems 7.6 (1999): 643-658.) states that:

**"If more than one combination of $\theta_i$ gives us the same point in the centroid, we keep the one with the largest membership grade."**

In this case, therefore, we should have kept only one of the two values, thus resulting in a  $O{C_{\tilde{G}_D}} = 5.14714$

#### Code and Results

This example can be implemented using the T2FUZZ-Library using the following code:

```python
from type2fuzzy import GeneralType2FuzzySet
from type2fuzzy import gt2_mendeljohn_reduce
from type2fuzzy import cog_defuzzify

set_definition = ''' (0.33/0.0)/1 + (0.66/0)/2 +
    (0.66/0+0.66/0.5)/3 +(0.33/0 + 0.66/0.5)/4 +
    (1/1)/5 + (0.33/0.5)/6 +(0.33/0 + 1/0.5)/7 +
    (0.66/0)/8 + (0.66/0)/9'''

gt2fs = GeneralType2FuzzySet.from_representation(set_definition)

print(gt2fs)

centroid = gt2_mendeljohn_reduce(gt2fs)
print(centroid)

defuzz = cog_defuzzify(centroid)
print(defuzz)
```

That yields the following results:

$$\tilde{G}_D = $$
$$(0.3300 / 0.0000) / 1.0000 + (0.6600 / 0.0000) / 2.0000 +$$
$$(0.6600 / 0.0000 + 0.6600 / 0.5000) / 3.0000  + $$
$$(0.3300 / 0.0000 + 0.6600 / 0.5000) / 4.0000 +$$
$$(1.0000 / 1.0000) / 5.0000 + (0.3300 / 0.5000) / 6.0000 +$$
$$(0.3300 / 0.0000 + 1.0000 / 0.5000) / 7.0000 +$$
$$(0.6600 / 0.0000) / 8.0000 + (0.6600 / 0.0000) / 9.0000$$$$

which is the general type-2 fuzzy set that is used

$$C_{\tilde{G}_D} = 0.330/5.333 + 0.330/5.750 + $$
$$0.330/5.000 + 0.330/5.400 + 0.330/4.750 + 0.330/5.200 + 0.330/4.600$$

this is the result obtained in the paper, except that there is only one entry for $x=5$

$$O{C_{\tilde{G}_D}} = 5.147619047619048$$, as calculated previously

### Computing the centroid of a General Type-2 Fuzzy Set by combining the Centroids of its zSlices

The original general type-2 fuzzy set used in this paper contains three zSlices

$$\tilde{Z}_{1} = \frac{1}{3}$$

$$\tilde{Z}_{2} = \frac{2}{3}$$

$$\tilde{Z}_{3} = 1$$

The authors did not use $\tilde{Z}_{0}$ (= 0) as it will not have any effect on the crisp centroid.

The membership values for the  upper and lower membership functions of each zSlice were summarized in the following table:

![zSlices Table](/post/img/type2fuzzy_paper_zslices_img1.JPG)

The KM iterative procedure was applied to determine the centroid at each zSlice, obtaining the following results:

|  |  |
|---|---|
|$\tilde{Z}_{1}$  = | [3.8656, 6.3$\bar{3}$] |
|$\tilde{Z}_{2}$  = |  [4.5454, 5.8628]|
|$\tilde{Z}_{3}$  = |  [5.3343, 5.3343]|

thus resulting in a centroid,

$$C_{\tilde{Z}} = \frac{1}{3} / [3.8656, 6.3\bar{3}] + \frac{2}{3} / [4.5454, 5.8628]+ 1 / [5.3343, 5.3343]$$

A centroid defuzzifier of the form,

\begin{equation}
y_c= \frac{(z_1(y_{l_1} + y_{r_1})/2) + \dots + (z_1(y_{l_N} + y_{r_N})/2)}{z_1+ \dots + z_N }
\end{equation}

resulting in a defuzzified crisp value $O_{C_{\tilde{Z}}} = 5.25176$

#### Code and Results

This example can be implemented using the T2FUZZ-Library using the following code. A general type-2 fuzzy set was constructed so that it will generate the same zSlices as specified in the paper.

```python
from type2fuzzy import GeneralType2FuzzySet
from type2fuzzy import ZSliceType2FuzzySet
from type2fuzzy import zslice_hagras_reduce
from type2fuzzy import zslice_centroid_defuzzify

set_definition ='''
(0.34 / 0.2 + 0.34 / 0.10 + 0.34 / 0) / 1+

(0.34 / 0.4 + 0.34 / 0.30 + 0.67 / 0.25 + 
0.67 / 0.20 + 0.67 / 0.10 + 0.67 / 0.00 )/2+

(0.34 / 0.6 + 0.67 / 0.50 + 0.67 / 0.40 + 
0.67 / 0.30 + 1.00 / 0.33 + 0.67 / 0.2 + 0.67 / 0.1 + 0.67 / 0 )/3+

(0.34 / 0.8 + 0.67 / 0.75 + 0.67 / 0.70 + 1.00 / 0.67 + 0.67 / 0.60 + 
0.67 / 0.5 + 0.33 / 0.4 + 0.33 / 0.3 + 0.34 / 0.2 + 0.34 / 0.1 + 
0.34 / 0.00)/4+

(1.00 / 1.00)/5+

(0.34 / 0.8 + 0.67 / 0.78 + 1.00 / 0.75 + 0.67 / 0.70 + 0.67 / 0.6 + 
0.34 / 0.5 + 0.34 / 0.4 + 0.34 / 0.33 )/6+

(0.34 / 0.6 + 0.67 / 0.56 + 1.00 / 0.50 + 0.67 / 0.40 + 
0.67 / 0.3 + 0.67 / 0.2 + 0.34 / 0.1 + 0.34 / 0.00)/7+

(0.34 / 0.4 + 0.67 / 0.33 + 0.67 / 0.30 + 1.00 / 0.25 + 
0.67 / 0.2 + 0.67 / 0.1 + 0.67 / 0.0)/8+

(0.34 / 0.2 + 0.67 / 0.11 + 0.67 / 0.10 + 1.00 / 0.00 )/9
'''

# create a general type-2 fuzzy set from the definition
gt2fs = GeneralType2FuzzySet.from_representation(set_definition)
print(f'Number of Embedded Sets: {gt2fs.embedded_type2_sets_count()}')

# generate a zslice type-2 fuzzy set fro a general type-2 set
# specify the number opf slices
zt2fs = ZSliceType2FuzzySet.from_general_type2_set(gt2fs, 3)
print(f'zSlices: \n {zt2fs}')

# reduce the zSlice type-2 set
z_centroid = zslice_hagras_reduce(zt2fs)
print(f'zSlices Centroid \n {z_centroid}')

# defuzzify the zSlice centroid
z_defuzz = zslice_centroid_defuzzify(z_centroid)
print(f'Crisp Centoid: {z_defuzz}')
```

```text
Number of Embedded Sets: 2838528
```

The general type-2 set yields a larger number of embedded type-2 set than that used by the authors.

```text
zSlices
-------

slice 0.0:
[0.00000, 0.20000]/1.0
+[0.00000, 0.40000]/2.0
+[0.00000, 0.60000]/3.0
+[0.00000, 0.80000]/4.0
+[1.00000]/5.0
+[0.33000, 0.80000]/6.0
+[0.00000, 0.60000]/7.0
+[0.00000, 0.40000]/8.0
+[0.00000, 0.20000]/9.0

slice 0.3333333333333333:
[0.00000, 0.20000]/1.0
+[0.00000, 0.40000]/2.0
+[0.00000, 0.60000]/3.0
+[0.00000, 0.80000]/4.0
+[1.00000]/5.0
+[0.33000, 0.80000]/6.0
+[0.00000, 0.60000]/7.0
+[0.00000, 0.40000]/8.0
+[0.00000, 0.20000]/9.0

slice 0.6666666666666666:
[0.00000, 0.25000]/2.0
+[0.00000, 0.50000]/3.0
+[0.50000, 0.75000]/4.0
+[1.00000]/5.0
+[0.60000, 0.78000]/6.0
+[0.20000, 0.56000]/7.0
+[0.00000, 0.33000]/8.0
+[0.00000, 0.11000]/9.0

slice 1.0:
[0.33000]/3.0
+[0.67000]/4.0
+[1.00000]/5.0
+[0.75000]/6.0
+[0.50000]/7.0
+[0.25000]/8.0
+[0.00000]/9.0
```

zSlices Centroid}, $C_{\tilde{Z}}$

```text
0.0 : [3.86561, 6.39526]
0.3333333333333333 : [3.86561, 6.39526]
0.6666666666666666 : [4.54545, 5.86280]
1.0 : [5.33429]
```

**Crisp Centroid**, $O_{C_{\tilde{Z}}}$: 5.256925833333334
