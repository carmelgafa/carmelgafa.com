---
title: The R Notes - Part 2
description: Part 2 of notes about the R language
date: "2020-02-08T16:15:48+01:00"
publishDate: "2020-02-08T16:15:48+01:00"
---

## Factors

Factors are used to store categorical data, ordered or unordered. 
Can be viewed as an integer vector where each entry has a label, better that categorizing with integers.

```R
x<- factor(c("low", "high", "high", "high", "low", "low", "high", "high"))
print(x)
```

will store the values in x. It will also tell you the labels in the list:

```
[1] low  high high high low  low  high high
Levels: high low
```

Calling the table function will tell you the frequency of the tables

```R
print(table(x))
```

```
high  low 
   5    3 
```

**unclass** method will trip the labels from the factor:

```R
print(unclass(x))
```

```
[1] 2 1 1 1 2 2 1 1
attr(,"levels")
[1] "high" "low"
```

notice that it is an integer vector
order of labels can be set using the levels argument

```R
x<- factor(c("low", "high", "high", "high", "low", "low", "high", "high"))
print(x)

y<- factor(c("low", "high", "high", "high", "low", "low", "high", "high"),levels = c("low", "high"))
print(y)
```

```
[1] low  high high high low  low  high high
Levels: high low
[1] low  high high high low  low  high high
Levels: low high
```

## NaN and NA

NaN: undefined mathematical operations
NA: The rest

to detect use .isnan() or .isna()


## Names
Objects can have names to describe them. Consider

```R
x<-1:3
print(x)

print(names(x))
```

```
[1] 1 2 3
NULL
```

### Assigning names:

```R
names(x)<- c("stephen", "colin", "mark")
print(x)
```

```
stephen   colin    mark 
      1       2       3 
```

### Manipulating elements:

```R
print(x["stephen"]+7)

print(x[1]+7)
```

```
stephen 
      8 

stephen 
      8 
```

### Naming elements in Lists:

```R
y<-list("stephen" = 12, "colin" = TRUE, "mark" = "hello")

print(y)
```


```
$stephen
[1] 12

$colin
[1] TRUE

$mark
[1] "hello"
```


Elements in Matrices can be named using the dimnames() function

```R
dimnames(m)<- list(c("Joe", "Peter"), c("Maths", "English", "Physics"))
print(m)

print(m["Peter", "English"])
```


```
      Maths English Physics
Joe       1       3       5
Peter     2       4       6

[1] 4
```
