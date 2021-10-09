---
title: "Introduction to type-2 fuzzy sets"
date: 2021-10-07T12:01:07+01:00
tags: [type-2, fuzzy, set]
draft: true
---
In a previous post, we have seen how we use a type-1 fuzzy set when we cannot determine the membership of an element as 0 or 1. Similarly, when the circumstances are so fuzzy that we have trouble determining the membership grade even as a crisp number in [0, 1], we use a type-2 fuzzy set.

This post will look at the basic concepts behind type-2 fuzzy sets and how we can represent a type-2 set using the python type-2 fuzzy library. We will base this discussion on "Type-2 Fuzzy Sets made Simple" by Robert John and Jerry Mendel, possibly the best paper to learn about type-2 fuzzy sets and logic.

### What is a type-2 fuzzy set?

Type-2 fuzzy logic is motivated by the premise that concepts have different meanings to different people. If we, for example, had to ask two different persons to plot their perception of the fuzzy set hot in the temperature universe of discourse and constrain them to use a trapezoidal set for their description, almost certainly we will get different definitions.

![Difference in type_1](/post/img/type_2_example_1.png)





A type-2 fuzzy set A, ( $ \tilde{A} $ ) can be 




![Difference in type_2](/post/img/type_2_set.jpg)