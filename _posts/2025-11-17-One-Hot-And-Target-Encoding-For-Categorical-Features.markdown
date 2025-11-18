---
layout: post
title:  "Working with Categorical Features with One Hot Encoding and Target Encoding"
date:   2025-11-17 14:14:11 -0700
tags: [data-science, python] 
---

Melbourne Housing Dataset is a common dataset every Kaggler goes through when they start with the introductory machine learning course. But the dataset has one issue that is not addressed in the course, and with good reasons, and that is too many categorical features. Handling categorical features is taught in the Intermediate course.

This dataset has so many categorical features and some of the features have too many categories. And in the introductory course, only numerical features are used to keep things simple. So, in this notebook, I've demonstrated the use of One Hot Encoding and Target Encoding for other categorical features.

Objective
The objective of this notebook is to demonstrate how One Hot Encoding and Target Encoding can be used to convert Categorical features to Numerical features so that they can be further used for other purposes to make our model better.

The objective in this notebook is not to get the best result. Getting best result will take a lot more than encoding, and more after encoding(for eg. checking correlations, smoothing, tuning hyperparameters etc.).

{% highlight python %}

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

{% endhighlight %}