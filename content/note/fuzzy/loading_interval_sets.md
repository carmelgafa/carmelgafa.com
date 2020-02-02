---
title: Loading Interval Type-2 Sets
description: How to load an IT2FS using the Type2 Fuzzy Library
date: "2020-01-12T15:09:43+01:00"
publishDate: "2020-01-12T15:09:43+01:00"
---

Interval Type-2 Fuzzy Sets can be loaded by using one of the following methods:

1. From a set definition in a string.
2. From a set definition in a file.

The set definitions myst have the following format:

```
[0.1,	0.5]/1 + 
[0.2,	0.7]/2 + 
[0.3,	1.0]/3 +
[0.4,	1]/4
```

The following example illustrated the creation of IT2FS using these methods

```
from type2fuzzy import IntervalType2FuzzySet
from type2fuzzy import it2_kernikmendel_reduce
import os


# load an it2fs from representation
set_representation= '''[0.1, 0.5]/1 + [0.2, 0.7]/2 + [0.3, 1.0]/3 + [0.4, 1]/4'''

it2fs = IntervalType2FuzzySet.from_representation(set_representation)

print(it2fs)


# load an it2fs from file
it2fs2 = IntervalType2FuzzySet.load_file(os.path.join(os.path.dirname(__file__),'test_it2fs.txt'))

print(it2fs2)
```

In the latter example, the following [file](/note/fuzzy/test_it2fs.txt) was used.