---
title: The R Notes - Part 1
description: Part 1 of notes about the R language
date: "2020-02-08T16:00:59+01:00"
publishDate: "2020-02-08T16:00:59+01:00"
---

Some quick and dirty notes on the R language:

Execute Script from Console
source("xxxx.r")

## Assign to Variable and Printing
use <- to assign, comments following #

``` R
x<-1
print(x)

y<-1:10
print(y)

#hello message
msg<-"hello"
print(msg)
```

## Vectors and Lists

Using Vector:

```R
x<- vector()
x[1] <- 2
x[2] <- 5
print(x)
```

will produce:
```[1] 2 5```

Storing items of different types, use List

```R
#using list
y<- list()
y[1] <- 3
y[2] <- "hello"
y[3] <- 'w'

print(y)
```
Will produce

```
[[1]]
[1] 3

[[2]]
[1] "hello"

[[3]]
[1] "w"
```

You can get attributes of collections

```R
print(class(x))
print(length(x))
```

will produce:

```
[1] "numeric"
[1] 2
```


## Matrices

### Matrix creation

```R
m <- matrix(nrow = 5, ncol = 10)
print(m)
```

will produce:

```
     [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
[1,]   NA   NA   NA   NA   NA   NA   NA   NA   NA    NA
[2,]   NA   NA   NA   NA   NA   NA   NA   NA   NA    NA
[3,]   NA   NA   NA   NA   NA   NA   NA   NA   NA    NA
[4,]   NA   NA   NA   NA   NA   NA   NA   NA   NA    NA
[5,]   NA   NA   NA   NA   NA   NA   NA   NA   NA    NA
```

You can get the dimensions of the matrix,

```R
print(dim(m))
```

will give 

```[1]  5 10```


### Changing matrix element

```R
m[2,3]<-4
print(m)
```

will produce

```
     [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
[1,]   NA   NA   NA   NA   NA   NA   NA   NA   NA    NA
[2,]   NA   NA    4   NA   NA   NA   NA   NA   NA    NA
[3,]   NA   NA   NA   NA   NA   NA   NA   NA   NA    NA
[4,]   NA   NA   NA   NA   NA   NA   NA   NA   NA    NA
[5,]   NA   NA   NA   NA   NA   NA   NA   NA   NA    NA
```

### Changing a whole row and column

```R
m[3,]<-2:11
print(m)

m[,10]<-1:5
print(m)
```

will produce:

```
[1,]   NA   NA   NA   NA   NA   NA   NA   NA   NA    NA
[2,]   NA   NA    4   NA   NA   NA   NA   NA   NA    NA
[3,]    2    3    4    5    6    7    8    9   10    11
[4,]   NA   NA   NA   NA   NA   NA   NA   NA   NA    NA
[5,]   NA   NA   NA   NA   NA   NA   NA   NA   NA    NA
```

```
     [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
[1,]   NA   NA   NA   NA   NA   NA   NA   NA   NA     1
[2,]   NA   NA    4   NA   NA   NA   NA   NA   NA     2
[3,]    2    3    4    5    6    7    8    9   10     3
[4,]   NA   NA   NA   NA   NA   NA   NA   NA   NA     4
[5,]   NA   NA   NA   NA   NA   NA   NA   NA   NA     5
```


### Assigning all elements of the matrix:

```R
m[,]<-1:50
print(m)
```

```
      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
[1,]    1    6   11   16   21   26   31   36   41    46
[2,]    2    7   12   17   22   27   32   37   42    47
[3,]    3    8   13   18   23   28   33   38   43    48
[4,]    4    9   14   19   24   29   34   39   44    49
[5,]    5   10   15   20   25   30   35   40   45    50
```

### Transforming vector to matrix

```R
v=1:30
print(v)

dim(v)<- c(5,6)
print(v)
```

```
 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
 26 27 28 29 30
```

```
     [,1] [,2] [,3] [,4] [,5] [,6]
[1,]    1    6   11   16   21   26
[2,]    2    7   12   17   22   27
[3,]    3    8   13   18   23   28
[4,]    4    9   14   19   24   29
[5,]    5   10   15   20   25   30
```

### Binding vectors

```R
a<-1:10
b<-11:15
c<-21:30

print(cbind(a,b,c))
print(rbind(a,b,c))
```

```
       a  b  c
 [1,]  1 11 21
 [2,]  2 12 22
 [3,]  3 13 23
 [4,]  4 14 24
 [5,]  5 15 25
 [6,]  6 11 26
 [7,]  7 12 27
 [8,]  8 13 28
 [9,]  9 14 29
[10,] 10 15 30
```

```
  [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
a    1    2    3    4    5    6    7    8    9    10
b   11   12   13   14   15   11   12   13   14    15
c   21   22   23   24   25   26   27   28   29    30
```
