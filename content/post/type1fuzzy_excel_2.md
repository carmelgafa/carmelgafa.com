---
title: "Fuzzy Control System In Excel version 1.2"
date: 2021-12-13T08:11:47+01:00
draft: false
tags: [type1fuzzy, excel]
---

I have received some questions regarding the [Fuzzy System in Excel](https://carmelgafa.com/post/type1fuzzy_excel/) that I posted some time ago, so here I am going through a slightly improvedc version that you can download from [Github](https://github.com/carmelgafa/type1fuzzy_excel.git).

This system implemented in this workbook illustrates a two-input / single output fuzzy system with two rules in an excel sheet.

- Sheets x1 and x2 define the inputs to the system. Each input variable consists of three triangular S, M, and L sets. You can adjust the membership of the sets by changing the Low, Medium and High parameters for each set.

- Similarly, sheet y defines the system's output, which consists of three triangular S, M, and L sets. You can adjust the membership of the sets by changing the Low, Medium and High parameters for each set.

- Two sheets define the rules for the system:
"if (x1 is S) and (x2 is M) then y is S" and "if (x1 is M) and (x2 is L) then y is M". The sheets show the result of the rules by
    - comparing the degree of belonging of the subsequent clauses and noting the minimum value
    - working out the rule's output using this value and the set specified in the consequent clause.

You can enter new input values in the Calculation tab and observe the calculated output. This sheet also shows the defuzzification calculation to obtain the system output.
