---
title: "Introduction to type-2 fuzzy sets"
date: 2021-10-15
tags: [type2-fuzzy, fuzzy]
draft: false
---
This article was first published in [Towards Data Science](https://towardsdatascience.com/type-2-fuzzy-sets-812c5b4d602d#d16c-9d2b4dddac26).

### Introduction

In a previous post, we have seen how we use a fuzzy set of type-1  when we cannot determine the membership of an element as 0 or 1. We can extend this concept when the circumstances are so fuzzy that we have trouble determining the membership grade as a number in [0, 1]. In these cases, type-2 fuzzy sets provide the necessary framework to formalize and work with this information.

This post will look at the basic concepts behind type-2 fuzzy sets. We will base this discussion on "Type-2 Fuzzy Sets made Simple" by Robert John and Jerry Mendel, possibly the best paper to learn about type-2 fuzzy sets and logic.

### What is a type-2 fuzzy set?

We start this discussion by restating the motivation behind the fuzzy set theory.  Let us suppose that today's temperature is 25 degrees Celcius, or 77 Ferenheight. Can we consider today a hot day? What does a precise thermometer reading tell us about the day, and how should we regulate our behaviour based on this reading?

The US National Oceanic and Atmospheric Administration (NOAA) considers the range between 26 and 32 degrees as 'caution' and identifies ranges further up the temperature scale as 'dangerous' and 'extremely dangerous.' If we adhere to this definition, we can imagine the bounds of a hot day drawn by NOAA. Therefore, we can conclude that today will not fall in the category of 'hot' days, maybe in the 'warm' or 'slightly hot' one. 

Categorizing a day using this logic will mean that a temperature of anything less than 26 and more than 32 degrees will not make the day 'hot'.

If we view the concept of a 'hot' day from a fuzzy lens, our reasoning will change. We can arguably consider 20 degrees a bit 'hot', while 30 degrees as positively 'hot'. On the other hand, we can classify forty degrees as a temperature beyond 'hot'; maybe it can fall under the 'torrid' category. Viewing 'hot' as a fuzzy concept will transform the rigid boundary into a curve that defines different grades of belonging for every temperature in the various categories. Thus a temperature can be 'warm'. 'hot' and 'torrid' simultaneously, albeit to different degrees in each category.

![type-1 fuzzy set](/post/img/type2_discrete_to_type1.png)

 If we now ask two different persons to plot their perception of the fuzzy set 'hot' and constrain them to use a trapezoidal set for their description, almost certainly, we will get two slightly different definitions. We will see that the degree of belonging that a given temperature has to the set 'hot' will be progressively smudged as we ask more people to submit their definition, and every point describing a degree of belonging will, in turn, transform into a fuzzy set in a three-dimensional function. The result is a fuzzy set of type-2.

Type-2 fuzzy logic is therefore motivated by the premise that concepts have different meanings to different people.

![type-2 set motivation](/post/img/type_2_example_1.png)

A type-2 fuzzy set for defining the concept of 'hot' temperature that we will denote by A, ( $ \tilde{A} $ ) can be as depicted below:

![type_2 fuzzy set](/post/img/type2_set.png)

### Type-2 fuzzy Sets

The membership function of a Type-2 Fuzzy Set is three dimensional, with the 

- x-axis called the **primary variable**
- the y-axis is called the **secondary variable** or secondary domain denoted by $u$. We note that this axis represents the degree of belonging in type-1 sets and therefore ranges in $[0,1]$. However, we have a range of degree of belonging values for every primary variable value in this case.
- the z-axis called the **membership function value** ( or secondary grade) that is denoted  by $ \mu$.

The most straightforward way to define a type-2 fuzzy set is to consider the collection of all the points in the three-dimensional space that make up a set. Formally, therefore, $\tilde{A}$ can also be expressed as:

$$\tilde{A}=\int_{x\in X}\int_{u\in J_{x}} \mu_{\tilde{A}}(x,u) / (x,u)$$

where $\int\int$ is the union over admissible $x$ and $u$ for a continuous universe of discourse (for discrete universes of discourse use $\sum\sum$ instead),

We can therefore describe the  type-2 set shown above as follows:

```{math}
(1 / 0 + 0.3 / 0.2)/15 +
(0.6 / 0.2 + 1 / 0.4 +0.7 / 0.6 )/20 +
(0.4 / 0.6 + 0.7 / 0.8 +1 / 1 )/25 +
(0.4 / 0.6 + 0.7 / 0.8 + 1 / 1 ) /30 +
(0.2 / 0.3 + 0.6 / 0.5 + 1 / 0.7 +0.4 / 1 ) / 35 +
(0.3 / 0.2 + 1 / 0.4 + 0.4 / 0.6  ) /40 +
(1 / 0 + 0.6 / 0.2 ) / 45
```

Furthermore, $J_{x}\subseteq[0,1]$ is known as the primary membership and will be discussed in more detail in the next section.

#### Vertical Slices

If we isolate a primary variable value, we notice that we obtain a type-1 fuzzy set called a vertical slice.

![type_2 fuzzy set vertical slice](/post/img/type2_verticalslice_1a.png)

![type_2 fuzzy set vertical slice](/post/img/type2_verticalslice_2.png)

A vertical slice is therefore a Type-1 fuzzy set $\mu_{\tilde{A}}(x=x',u)$ for $x\in X$ and $\forall u \in J_{x'}\subseteq[0,1]$, that formally defined as:

$$\mu_{\tilde{A}}(x=x',u)=\int_{u\in J_{x'}}f_{x'}(u) / u$$

where $0\leq f_{x'}(u)\leq 1$

The function resulting from a vertical slice is also referred to as a **secondary membership function**, written formally as  $\mu_{ \tilde{A} }(x')$, that is $\mu_{ \tilde{A} }(x=x', u)$ for $x \in X$ and $\forall u \in J_x \subseteq [0, 1]$

$$\mu_{ \tilde{A} }(x=x' , u) \equiv \mu_{ \tilde{A} }(x') = \int_{u \in J_{x'}} f_{x'}(u)/u ; J_{x'}\subseteq[0,1]$$

where $0 \leq f_{x'}(u) \leq 1 $

The secondary membership function at $x=20$ is, therefore:

```{math}
0.6 / 0.2 + 1 / 0.4 +0.7 / 0.6 
```

The **domain** of a secondary membership function is called the **primary membership** of $x$. Hence in

$$\tilde{A}=\int_{x \in X} \mu_{ \tilde{A} }(x) /x = \int_{x \in X} \left[  \int_{u\in J_{x}} f(u) / u \right]  /x$$

As mentioned above, $J_{x}$ is also known as the **primary membership function**, where $J_{x} \subseteq [0,1]$ for $\forall x \in X$

```{math}
J_{20}= {.2, 0.4, 0.6}
```

The **amplitude** of a secondary membership function is the **secondary grade**. Hence in

$$\tilde{A}=\int_{x \in X} \mu_{ \tilde{A} }(x) /x = \int_{x \in X} \left[  \int_{u\in J_{x}} f(u) / u \right]  /x$$

where $J_{x} \subseteq [0,1]$, $f(u)$ is the secondary grade.

If $X$ and $J_{x}$ are discrete:

$$\tilde{A} = \sum_{x \in X} \left[ \sum_{u \in J_{x}} f(u) / u \right] /x$$

Regarding secondary grades, we can note that in

$$\tilde{A} = \{(x, u), \mu_{ \tilde{A} }(x, u)|\forall x\in X, \forall u \in J_{x} \subseteq [0,1]\}$$

$\mu(x', u')(x \in X, u' \in J_{x'})$ is a secondary grade.

#### Embedded Type-2 Sets

For discrete  $X$ and $U$, an **embedded type-2 set**  can be created by taking a single element out of each secondary membership function in a set. Hence, if $X$ has $N$ elements, $\tilde{A_e}$ has exactly one element from $J_{x_{1}}, J_{x_{2}}, \dots , J_{x_{N}}$; namely $u_{1}, u_{2}, \dots , u_{N}$ each with associated grade namely $f_{x_{1}}(u_1), f_{x_{2}}(u_2), \dots , f_{x_{N}}(u_N)$, such that:

![embedded type_2 fuzzy set](/post/img/type2_embedded_b.png)

$$\tilde{A_e} = \displaystyle \sum_{i=1}^{N} \left[ f_{x_{i}} (u_{i}) \right] / x_{i}$$

where $u_{i} \in J_{x_{i}} \subseteq [0,1]$

There are a total of: $ \displaystyle \prod_{i=1}^{N} M_i$ embedded sets in $\tilde{A}$

The identified embedded type-2 fuzzy set, $\tilde{A_e}^1$, is therefore:

```{math}
1 / 0 / 15 + 0.6 / 0.2 / 20 + 0.4 / 0.6 / 25 + 0.4 / 0.6 /30 + 0.2 / 0.3 / 35 + 0.3 / 0.2 /40 + 1 / 0 / 45
```

We also notice that our set contains

```{math}
2 x 3 x 3 x 3 x 4 x 3 x 2 = 1296
```

,embedded type-2 sets.

We can also then represent a  type-2 fuzzy set by the collection of all the embedded type-2 sets, known as the **wavy-slice representation** or the **Mendel-John representation**. The wavy-slice representation is, therefore:

$$\tilde{A}=\bigcup_{\forall j} \tilde{A}_{e}^{j}$$

#### Type-1 fuzzy sets

A type-1 fuzzy set can be represented as a type-2 fuzzy set. Its type-2 representation is: 
$ \left( 1/ \mu(x) \right) / x $ or $ 1/\mu_{F}(x), \forall x\in X$.

$1/ \mu_{F}(x)$ means that the secondary membership function has only one value in its domain, i.e. the primary membership $\mu_F(x)$ at which the secondary grade is equal to 1. 

#### Footprint of Uncertainty

The 2D support of $\mu$ is called the **footprint of uncertainty (FOU)**

$$FOU(\tilde{A})= \left\lbrace  (x,u) \in X \times [0,1]  | \mu_{\tilde{A}}(x,u) > 0 \right\rbrace $$

FOU represents the uncertainty in the primary memberships of $\tilde{A}$. It is the union of all primary memberships.

$$FOU(\tilde{A}) = \bigcup\limits_{x\in X} J_{x}$$

![footprint of uncertainty](/post/img/type2_fou.png)

The shaded FOU implies a distribution at the top of the type-2 fuzzy set in the third dimension that depends on the choice of the secondary grades. When all the secondary grades are of a type-2 set are equal to 1, the set is called an **interval type-2 fuzzy set**.

We notice that the footprint of uncertainty has upper and lower bound, referred to as the upper and lower membership functions:

The **lower membership function** 

$$LMF(\tilde{A}) = \underline{\mu_{\tilde{A}}} = \inf\left\lbrace u | u\in[0,1], \mu_{ \tilde{A} }(x,u) >0 \right\rbrace $$

![lower membership function](/post/img/type2_fou_lmf.png)

The **upper membership function**

$$LMF(\tilde{A}) = \overline{\mu_{\tilde{A}}} = \sup\left\lbrace u | u\in[0,1], \mu_{ \tilde{A} }(x,u) >0 \right\rbrace $$

![upper membership function](/post/img/type2_fou_umf.png)
