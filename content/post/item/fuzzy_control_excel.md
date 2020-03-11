---
title: Fuzzy Control System In Excel
description: An implementation of a fuzzy control system in Excel
thumbnail: 
tags: [fuzzy]
draft:
summary: An implementation of a fuzzy control system in Excel
date: "2020-03-11T18:30:40+01:00"
publishDate: "2020-03-11T18:30:40+01:00"
---

I am currently extending my [Type-2 Fuzzy Logic Library](/portfolio/type2fuzzylibrary/type2fuzzylibrary) so that  fuzzy controllers can be implemented through its codebase. The implementation is straightforward if you have the correct structures in place; using an inference system, the most popular of which was discussed by Mamdani (Mamdani, Ebrahim H. "Application of fuzzy algorithms for control of simple dynamic plant." Proceedings of the institution of electrical engineers. Vol. 121. No. 12. IET, 1974.). An illustration of the Mamdani inference system operation is below (taken from wikipedia):

![Mamdani FIS](/post/img/fuzzy_inference.jpg)

Testing such a system can be a challenge and for this reason I have created a version of Mamdani FIS in Excel. The structure is quite simple; the workbook implements a 2-input, single output system with 2 rules that can be easily extended if necessary. The inputs and outputs contain 3 fuzzy sets each. An 'input and results' sheet provides a mechanism so that the discrete inputs are entered for each variable and the resulting fuzzy output can be observed.

The sheet can be downloaded [here](/post/files/fuzzy_system.xlsx)