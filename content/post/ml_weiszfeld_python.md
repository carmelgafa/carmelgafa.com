---
title: "Simple Python implementation of the Weiszfeld algorithm"
date: "2021-03-14T15:32:17+01:00"
draft: false
tags: [machine-learning, python, weiszfeld_algorithm]
---


Following is a simple implementation of the Weiszfeld algortihm that was discussed in a previous post in python.

```python
import numpy as np
import math
from numpy import array

def weiszfeld(points):

    max_error = 0.0000000001

    x=np.array([point[0] for point in  points])
    y=np.array([point[1] for point in  points])


    ext_condition = True

    start_x = np.average(x)
    start_y = np.average(y)

    while ext_condition:

        sod = (((x - start_x)**2) + ((y - start_y)**2))**0.5

        new_x = sum(x/sod) / sum(1/sod)
        new_y = sum(y/sod) / sum(1/sod)

        ext_condition = (abs(new_x - start_x) > max_error) or 
            (abs(new_y - start_y) > max_error)

        start_y = new_y
        start_x = new_x

        print(new_x, new_y)


if __name__=="__main__":
    weiszfeld([(2,1), (12,2), (3,9), (13,11)])
```
