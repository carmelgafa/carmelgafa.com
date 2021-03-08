---
title: Type-1 Fuzzy Variable
description: How to work with Type-1 fuzzy variables in Type2Fuzzy Library
date: "2020-03-08T14:21:53+01:00"
publishDate: "2020-03-08T14:21:53+01:00"
tags: []
---

The **Type1FuzzyVariable** class in the library is a way to define and use linguistic variables.

By a linguistic variable we mean a variable whose values are words or sentences in a natural or artificial language. For example, Age is a linguistic variable if its values are linguistic rather than numerical, i.e.,young, not young, very young, quite young, old, not very old and not very young, etc., rather than 20, 21,22, 23 (Zadeh, Lotfi Asker. "The concept of a linguistic variable and its application to approximate reasoning." Learning systems and intelligent robots. Springer, Boston, MA, 1974. 1-10.)

This concept can be expressed programmatically using the following:

```python
from type2fuzzy import Type1FuzzyVariable

# adding an age linguistic variable
var = Type1FuzzyVariable(0, 100, 100)

# add fuzzy sets
var.add_triangular('very young', 0, 0, 20)
var.add_triangular('young', 10, 20, 30)
var.add_triangular('adult', 20, 40, 60)
var.add_triangular('old', 50, 70, 90)
var.add_triangular('very old', 70, 100, 100)

# visualize sets
var.plot_variable()
```

The above code snippet will create the variable and sets and will produce the following image:

![Linguistic Variable 'age'](/note/fuzzy/img/fuzzy_variable.jpeg)

It is also very useful to have the ability to generate a number of fuzzy sets automatically in a linguistic variable, as this technique is sometimes used in ML applications. The function **generate_sets** takes a parameter n so that

```latex
$number_of_sets = (2\times n) + 1$
```
The following code shows how the generation function can be used to generate 7 sets:

```python
from type2fuzzy import Type1FuzzyVariable

# adding an age linguistic variable
var = Type1FuzzyVariable(0, 100, 100)

# generate (2*3)+1 = 7 sets
var.generate_sets(3)

var.plot_variable()
```

The following is obtained:

![Linguistic Variable 'age'](/note/fuzzy/img/fuzzy_variable_gen.jpeg)

