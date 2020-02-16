---
title: Type-2 Fuzzy Logic Library
description: A Type-2 fuzzy logic implementation
date: "2020-01-14T09:34:11+01:00"
jobDate: 2019
work: [fuzzy-logic, python]
techs: [python]
designs: []
thumbnail: type2fuzzylibrary/type2fuzzy.jpg
projectUrl: http://t2fuzz.com
---

A type-2 fuzzy logic library providing:

1. Ways to define and work with general type-2 fuzzy sets
2. Ways to define and work with interval type-2 fuzzy sets
3. Ways to generate z-sliced sets from general type-2 fuzzy sets
4. Functions to perform wavy-slice type-reduction (Mendel-John) on general type-2 fuzzy sets
5. Functions to perform interval type-2 reduction (Karnik-Mendel)
6. Functions to perform partial-centroid type-reduction on general type-2 fuzzy sets
7. Functions to perform defuzzification of type-1 fuzzy sets
8. Tools to measure the performance of algorithms
9. Tools to plot general, interval and z-sliced type-2 fuzzy sets and type-1 fuzzy sets. and more

All type2fuzzy wheels distributed on PyPI are BSD licensed.

Examples of how this library was used to work some famous type-2 fuzzy logic papers can be found [here](https://github.com/carmelgafa/type2fuzzy_examples).

## Website
http://t2fuzz.com

## Change History:
### version 0.1.34 - 18.11.2019

* Ability to create Interval Type-2 fuzzy sets having a gaussian function with fixed mean and fixed standard deviation as per Karnik and Mendel 1996
* Karnik, Nilesh N., and Jerry M. Mendel. "Introduction to type-2 fuzzy logic systems." 1998 IEEE International Conference on Fuzzy Systems Proceedings. IEEE World Congress on Computational Intelligence (Cat. No. 98CH36228). Vol. 2. IEEE, 1998. (http://t2fuzz.com/membership/generate_it2fs.html)
* An experimental way to define General Type-2 fuzzy sets through [horizontal slices](http://t2fuzz.com/membership/generate_gt2mf.html)

### version 0.1.33 - 15.11.2019

* Updated repo information

### version 0.1.32 - 15.11.2019

* Get [domain limits for a type-1 fuzzy set](http://t2fuzz.com/membership/type1fuzzyset.html)
version 0.1.31 - 12.11.2019
* Added library [website](http://t2fuzz.com)
* Convert a gt2fs into an it2fs - An it2fs can be [generated form a gt2fs by using from_general_type2_set](http://t2fuzz.com/membership/)
* Creation of it2fs as found in literature - Creation of it2fs as [specified by Karnik and Mendel](http://t2fuzz.com/membership/generate_it2fs.html)