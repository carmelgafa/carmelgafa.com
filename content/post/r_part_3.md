---
title: The R Notes - Part 3
description: Part 3 of notes about the R language
date: "2020-02-08T16:24:22+01:00"
publishDate: "2020-02-08T16:24:22+01:00"
tags: [r]
---

## Reading and Writing Data

Reading tabular data - read.table & read.csv. there return a DataFrame. Equivalent for writing is write.table

```R
read.table(
file, 
header = FALSE,                                  # has separator
sep = "",                                        # seperator example ","default is space
quote = "\"'",
dec = ".", 
numerals = c("allow.loss", "warn.loss", "no.loss"),
row.names, 
col.names, 
as.is = !stringsAsFactors,
na.strings = "NA", 
colClasses = NA,                                 # character vector indicating the class
                                                 # of each vector. not required
nrows = -1,                                      # number of rows. not required
skip = 0,                                        # skip from beginneing
check.names = TRUE, 
fill = !blank.lines.skip,
strip.white = FALSE, 
blank.lines.skip = TRUE,
comment.char = "#",                              # comment symbol. anything to the right is                                                   ignored
allowEscapes = FALSE, 
flush = FALSE,
stringsAsFactors = default.stringsAsFactors(),   # encode character variables as factors
fileEncoding = "", 
encoding = "unknown", 
text, 
skipNul = FALSE)
```

Consider the following data file:

```
Name, Surname, Age, Rating, Senior
Mark, Brown, 40, 1, TRUE
Colin, Smith, 41, 1, TRUE
Joe, Blogs, 27, 2, FALSE
#Joe2, Blogs, 27, 2, FALSE
```

Use the following to read, 

```R
data<-read.table("info.txt",header = TRUE, 
sep="," , 
colClasses = c("character", "character", "integer",                 "integer", 
"logical")

print(data)
```

```
Name   Surname Age Rating Senior
1      Mark      Brown  40      1   TRUE
2     Colin     Smith  41      1   TRUE
3     Joe  Blogs  27      2  FALSE
```

When Separator is "," you can use read.csv. Header is always equal to TRUE

ReadLines reads any file and returns a character vector. Equivalent for writing is WriteLines 

Source will read R code, also dget. Equivalent for writing is Dump and dput

load & unserialize will read binary objects. Equivalent for writing is save & serialize
