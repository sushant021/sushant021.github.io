---
layout: post
title:  "Why is It Necessary To Balance an Imbalanced Dataset"
date:   2025-11-17 14:14:11 -0700
tags: [data-science, python] 
---

An imbalanced dataset is kind of a dataset that is skewed towards one class. For example in a credit card transaction data, you may see 1 or 2 fraud transactions out of 1000s of transactions. So that dataset is highly skewed towards "legit" class. 


{% highlight ruby %}

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

{% endhighlight %}

