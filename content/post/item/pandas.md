---
title: Short notes about Pandas
description:
thumbnail: 
tags: []
draft: Summary of Pandas basics
summary: Summary of Pandas basics
date: "2020-03-30T13:21:20+02:00"
publishDate: "2020-03-30T13:21:20+02:00"
---

I am currently using a lot more the Pandas library to load data into a fuzzy learning framework. This note summarizes my learning points on the library. For the process of this exercise the [Wine Quality data set](http://www3.dsi.uminho.pt/pcortez/wine/) used by Cortes et all  will be used

## Pandas Terminology

**Series** is a one-dimensional NumPy-like array. You can put any data type in here, and perform vectorized operations on it. A series is also a dictionary. Usually, denoted with **s**.

**DataFrame** is a two-dimensional NumPy-like array. Again, any data type can be stuffed in here. Usually,  denoted as **df**.

**Index**  is what the data is "associated" by. So if you have date series data, like coronavirus new cases, generally the index is the date.

**Slicing** is selecting specific batches of data.
- Sort data in just any possible, fast.
- Move columns around, add new ones, and remove others
- Perform operations on data through

	- custom code or
	- Pandas built-in functions, like standard deviation, correlation, or moving averages for example.

-   Finally, Matplotlib can be used to display data. Pandas works seamlessly with Matplotlib including data sets that have dates.

## Manipulating Data
  
The data set under consideration has a  CSV format, or comma-separated variable, file type.
Pandas use the **read_** and **to_** prefix to read and write to several different sources. In the case of CSV files, the [```read_csv```](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)  method is used.
  
The following code example illustrates reading and column operations on the data set under consideration:

```python
import pandas as pd
from pandas import DataFrame
import os

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'data\winequality-red.csv')

df = pd.read_csv(filename, sep=';')
print(df.head())

df2 = df['chlorides']
print(df2.head())

df3 = df[['free sulfur dioxide', 'total sulfur dioxide']]
print(df3.head())
```

The data file was downloaded and placed in a folder called **data**. It is, therefore, necessary to set the correct location to ```read_csv```. Upon examining the data file, it was noticed that data was separated using a ';' so the separator parameter.

The example illustrates how single or multiple columns can be extracted from the original **df**.

### Renaming columns

Renaming is done with the ```rename()``` function. A warning can arise when renaming using this method.

```python
to_rename = {'fixed acidity':'fixed_acidity',
'volatile acidity':'volatile_acidity',
'citric acid':'citric_acid',
'residual sugar':'residual_sugar',
'free sulfur dioxide':'free_sulfur_dioxide',
'total sulfur dioxide':'total_sulfur_dioxide'
}

df.rename(columns=to_rename, inplace=True)
print(df.head())
```

### Filtering Data

The following code filters the items with __residual sugar__ value greater than 10.

```python
df4 = df[(df['residual_sugar'] > 10)]
print(df4)
```

### Creating new Columns

Below a new column called **sulphur_dioxide_difference** is created that contains the difference between **total_sulfur_dioxide** and **free_sulfur_dioxide**

```python
df['sulphur_dioxide_difference'] = 
  df['total_sulfur_dioxide'] - df['free_sulfur_dioxide']
print(df.head())
```

## Plotting

[Matplotlib](https://matplotlib.org/) is a comprehensive library for creating static, animated, and interactive visualizations in Python. Pandas has  **tight integration**  with matplotlib where  data can be displayed from a DataFrame using the ```plot()``` method.

```python
df[['total_sulfur_dioxide','free_sulfur_dioxide',
  'sulphur_dioxide_difference']][200:300].plot()
plt.show()
```

The example above plots 100 samples from the 200 to the 300 element.
