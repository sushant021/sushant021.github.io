---
layout: post
title:  "Working with Categorical Features with One Hot Encoding and Target Encoding"
date:   2025-11-17 14:14:11 -0700
tags: [data-science, python] 
---

In the world of data and machine learning, most models expect numerical inputs. But what if you have a column like `Color` with values `Red, Blue, Green`. How are you going to feed that to a regression model that only understands numbers ? That's where encoding comes in. Such columns or features are called Categorical features or Categorical data. You can use such categorical data in your models once you convert them to numbers. That is what One Hot Encoding and Target Encoding are for. They convert categorical data to numbers so that you can use them on your models. 

For this we will use Melbourne Housing Dataset from Kaggle : [Melbourne Housing Dataset](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot)


We will see how One Hot Encoding and Target Encoding can be used to convert Categorical features in this dataset to Numerical features so that they can be further used for other purposes to make our model better. And we will also see what's the differences between the two techniques and how they work. 

Keep in mind that the goal of this demonstration is not to achieve the best model performance. Getting optimal results requires more than just encoding—such as exploring feature correlations, applying smoothing techniques, tuning hyperparameters, and more.

Let’s walk through how to apply these encodings using the Melbourne Housing dataset.

{% highlight python %}

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

{% endhighlight %}

Load the dataset. 

{% highlight python %}

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
df = pd.read_csv(melbourne_file_path)

#let's get rid of empty rows first
df = df.dropna(axis=0)
# reseting the index helps to concat dataframes more easily. We'll need that later
df = df.reset_index(drop=True)
df.head()
{% endhighlight %}


{%highlight python %}

Suburb	Address	Rooms	Type	Price	Method	SellerG	Date	Distance	Postcode	...	Bathroom	Car	Landsize	BuildingArea	YearBuilt	CouncilArea	Lattitude	Longtitude	Regionname	Propertycount
0	Abbotsford	25 Bloomburg St	2	h	1035000.0	S	Biggin	4/02/2016	2.5	3067.0	...	1.0	0.0	156.0	79.0	1900.0	Yarra	-37.8079	144.9934	Northern Metropolitan	4019.0
1	Abbotsford	5 Charles St	3	h	1465000.0	SP	Biggin	4/03/2017	2.5	3067.0	...	2.0	0.0	134.0	150.0	1900.0	Yarra	-37.8093	144.9944	Northern Metropolitan	4019.0
2	Abbotsford	55a Park St	4	h	1600000.0	VB	Nelson	4/06/2016	2.5	3067.0	...	1.0	2.0	120.0	142.0	2014.0	Yarra	-37.8072	144.9941	Northern Metropolitan	4019.0
3	Abbotsford	124 Yarra St	3	h	1876000.0	S	Nelson	7/05/2016	2.5	3067.0	...	2.0	0.0	245.0	210.0	1910.0	Yarra	-37.8024	144.9993	Northern Metropolitan	4019.0
4	Abbotsford	98 Charles St	2	h	1636000.0	S	Nelson	8/10/2016	2.5	3067.0	...	1.0	2.0	256.0	107.0	1890.0	Yarra	-37.8060	144.9954	Northern Metropolitan	4019.0

{% endhighlight %}

Let's look at the columns and classify which ones are categorical.

{% highlight python %}
df.columns

#output
Index(['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG',
       'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
       'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude',
       'Longtitude', 'Regionname', 'Propertycount'],
      dtype='object')
{% endhighlight %}

From the list above, `Suburb, Address, Type, Method, SellerG, Date, Postcode, CouncilArea, Regionname` are categorical features.

`Postcode` only appears numerical. Technically it can be used in an ML model without an error but it would give you an inappropriate result. Because, it represents a category. It has no numerical significance.

<h5 class="fw-bold mt-5">
Working With Just The Numerical Features
</h5>

Let's see the result when we use just the basic numerical features that work with Random Forest. And later we will see whether adding those categorical features will actually improve our model or make it worse.

{% highlight python %}
features = ['Rooms','Distance','Bedroom2','Bathroom', 'Car','Landsize',
            'BuildingArea','Lattitude','Longtitude','Propertycount']
y = df.Price
X = df[features]
train_X, test_X, train_y, test_y = train_test_split(X,y, random_state =1)
forest_model = RandomForestRegressor(random_state=1, n_estimators = 100 )
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(test_X)
print(mean_absolute_error(test_y, melb_preds))

#output
175105.8469491838

{% endhighlight %}

Now, let's work with the categorical data and add them into our features list.

<h5 class="fw-bold mt-5">
One Hot Encoding
</h5>

In One Hot Encoding, the categories for any feature are converted into columns. For example, in our `Type` column, there are three categories, so this one column will be expanded into three columns. Each column represents a category. The rows are filled with 0's and 1's. The final result is a matrix of 0's and 1's where presence of each category is represented with 1.

One Hot Encoding is only good for features that have few categories. Imagine if the `Type` column had 20 categories. That would expand one column to 20 columns. With too many categories, we will be cursed with the **Curse of Dimensionality**. In our dataset, it is only good for `Type` and `Method`. All the others have too many categories.

Okay, let's start encoding. 

{% highlight python %}
# create the encoder
ohe = OneHotEncoder(handle_unknown = 'ignore')

# fit the encoder with the categorical data 
encoded_columns = ohe.fit_transform(df[['Type','Method']])

# after fitting the data to the encoder, we can view the categories inside Type and Method, and the result data
# the resulting data will make more sense after you change it into an actual dataframe which we'll do later
print(ohe.categories_,'\n',encoded_columns.toarray())

# we will use the categories names as column names to add to the dataframe
column_names = np.concatenate([ohe.categories_[0],ohe.categories_[1]],axis = 0)

# Now let's create the dataframe 
encoded_df = pd.DataFrame(encoded_columns.toarray(),columns = column_names)
encoded_df

#output

[array(['h', 't', 'u'], dtype=object), array(['PI', 'S', 'SA', 'SP', 'VB'], dtype=object)] 
 [[1. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 1. 0.]
 [1. 0. 0. ... 0. 0. 1.]
 ...
 [0. 0. 1. ... 0. 1. 0.]
 [0. 0. 1. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 1.]]
{% endhighlight %}

Let's concatenate this encoded dataframe to our original dataframe. 

{% highlight python %}
df = pd.concat([df,encoded_df],axis =1)
df.head()

#output

Suburb	Address	Rooms	Type	Price	Method	SellerG	Date	Distance	Postcode	...	Regionname	Propertycount	h	t	u	PI	S	SA	SP	VB
0	Abbotsford	25 Bloomburg St	2	h	1035000.0	S	Biggin	4/02/2016	2.5	3067.0	...	Northern Metropolitan	4019.0	1.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
1	Abbotsford	5 Charles St	3	h	1465000.0	SP	Biggin	4/03/2017	2.5	3067.0	...	Northern Metropolitan	4019.0	1.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0
2	Abbotsford	55a Park St	4	h	1600000.0	VB	Nelson	4/06/2016	2.5	3067.0	...	Northern Metropolitan	4019.0	1.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0
3	Abbotsford	124 Yarra St	3	h	1876000.0	S	Nelson	7/05/2016	2.5	3067.0	...	Northern Metropolitan	4019.0	1.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
4	Abbotsford	98 Charles St	2	h	1636000.0	S	Nelson	8/10/2016	2.5	3067.0	...	Northern Metropolitan	4019.0	1.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
{% endhighlight %}

We have successfully converted two categorical features to numerical features. But this process has also added 6 more columns. This is the Curse of Dimensionality. More categories would result in more columns. That is why One Hot Encoding is not suitable for features with large number of categories. It will enlarge the dataset by a lot. That is where Target Encoding comes in.

<h5 class="fw-bold mt-5">
Target Encoding
</h5>

With target encoding, we replace the categories with their respective mean values of target data. For example let's just take the feature `Suburb` right now and check mean values for all the categories inside.

{% highlight python %}
suburb_means = df.groupby('Suburb')['Price'].mean()
suburb_means

#output
Suburb
Abbotsford      1.125972e+06
Aberfeldie      1.365000e+06
Airport West    7.042941e+05
Albanvale       5.550000e+05
Albert Park     1.868783e+06
                    ...     
Wollert         5.625000e+05
Wyndham Vale    4.860000e+05
Yallambie       8.646786e+05
Yarra Glen      6.200000e+05
Yarraville      1.018802e+06
Name: Price, Length: 287, dtype: float64
{% endhighlight %}

These mean values will replace the actual categories in `Suburb` feature. For instance, `Abbotsford` will be replaced by `1.125972e+06`. Now, let's see the actual implementation.

{% highlight python %}
suburb_means = df.groupby('Suburb')['Price'].mean()
postcode_means = df.groupby('Postcode')['Price'].mean()
sellerg_means = df.groupby('SellerG')['Price'].mean()
council_area_means = df.groupby('CouncilArea')['Price'].mean()
region_name_means = df.groupby('Regionname')['Price'].mean()
df['Suburb'] = df['Suburb'].map(suburb_means)
df['Postcode'] = df['Postcode'].map(postcode_means)
df['SellerG'] = df['SellerG'].map(sellerg_means)
df['CouncilArea'] = df['CouncilArea'].map(council_area_means)
df['Regionname'] = df['Regionname'].map(region_name_means)
df.head()

#output
Suburb	Address	Rooms	Type	Price	Method	SellerG	Date	Distance	Postcode	...	Regionname	Propertycount	h	t	u	PI	S	SA	SP	VB
0	1.125972e+06	25 Bloomburg St	2	h	1035000.0	S	1.027889e+06	4/02/2016	2.5	1.125972e+06	...	872263.457929	4019.0	1.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
1	1.125972e+06	5 Charles St	3	h	1465000.0	SP	1.027889e+06	4/03/2017	2.5	1.125972e+06	...	872263.457929	4019.0	1.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0
2	1.125972e+06	55a Park St	4	h	1600000.0	VB	1.011728e+06	4/06/2016	2.5	1.125972e+06	...	872263.457929	4019.0	1.0	0.0	0.0	0.0	0.0	0.0	0.0	1.0
3	1.125972e+06	124 Yarra St	3	h	1876000.0	S	1.011728e+06	7/05/2016	2.5	1.125972e+06	...	872263.457929	4019.0	1.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
4	1.125972e+06	98 Charles St	2	h	1636000.0	S	1.011728e+06	8/10/2016	2.5	1.125972e+06	...	872263.457929	4019.0	1.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
{% endhighlight %}

We are almost done. Now we have two categorical features that we haven't touched, `Address` and `Date`. We will ignore the `Address` column because it just has way too many categories to address (pun intended). From `Date` column, we will extract the year and get how many years ago it was sold.


<h5 class="fw-bold mt-5">
Working With Dates
</h5>

Apart from the obvious `Date` column, there's also the column `YearBuilt`. We could leave `YearBuilt` because it does have numerical significance and works fine. But it would be more appropriate to extract how old the house exactly is and use that instead. So, we're doing that.

{% highlight python %}
from datetime import date

current_year = date.today().year

# converting the Date column from String Object to datetime datatype
df['Date'] = pd.to_datetime(df['Date'],dayfirst = True)

# how many years ago was it sold ? 
sold_years_ago = [current_year - date.year for date in df['Date']]

# replacing the Date column with sold_years_ago data
df['sold_years_ago'] = sold_years_ago


# getting how old the house is from YearBuilt column
years_old = [current_year - year for year in df.YearBuilt]
df['years_old'] = years_old
{% endhighlight %}


<h5 class="fw-bold mt-5">
Final Dataset
</h5>

We are done working with all the data. Now, let's check out the final dataset.

{% highlight python %}
print(df.head(),'\n', df.columns, '\n')

#output

         Suburb          Address  Rooms Type      Price Method       SellerG  \
0  1.125972e+06  25 Bloomburg St      2    h  1035000.0      S  1.027889e+06   
1  1.125972e+06     5 Charles St      3    h  1465000.0     SP  1.027889e+06   
2  1.125972e+06      55a Park St      4    h  1600000.0     VB  1.011728e+06   
3  1.125972e+06     124 Yarra St      3    h  1876000.0      S  1.011728e+06   
4  1.125972e+06    98 Charles St      2    h  1636000.0      S  1.011728e+06   

        Date  Distance      Postcode  ...    h    t    u   PI    S   SA   SP  \
0 2016-02-04       2.5  1.125972e+06  ...  1.0  0.0  0.0  0.0  1.0  0.0  0.0   
1 2017-03-04       2.5  1.125972e+06  ...  1.0  0.0  0.0  0.0  0.0  0.0  1.0   
2 2016-06-04       2.5  1.125972e+06  ...  1.0  0.0  0.0  0.0  0.0  0.0  0.0   
3 2016-05-07       2.5  1.125972e+06  ...  1.0  0.0  0.0  0.0  1.0  0.0  0.0   
4 2016-10-08       2.5  1.125972e+06  ...  1.0  0.0  0.0  0.0  1.0  0.0  0.0   

    VB  sold_years_ago  years_old  
0  0.0               7      123.0  
1  0.0               6      123.0  
2  1.0               7        9.0  
3  0.0               7      113.0  
4  0.0               7      133.0  

[5 rows x 31 columns] 
 Index(['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG',
       'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
       'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude',
       'Longtitude', 'Regionname', 'Propertycount', 'h', 't', 'u', 'PI', 'S',
       'SA', 'SP', 'VB', 'sold_years_ago', 'years_old'],
      dtype='object')

{% endhighlight %}

Looking at the columns, we don't need some of them anymore. We can drop `Address, Type, Method, Date` and `YearBuilt`.

{% highlight python %}
df = df.drop(['Address','Type','Method','Date','YearBuilt'],axis = 1)
{% endhighlight %}

<h5 class="fw-bold mt-5">
Using The Final Dataset
</h5>

Let's use all the data and see the result.

{% highlight python %}
# making training and testing dataset
X = df.loc[:,df.columns != 'Price']
y = df.Price
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 1)
reg = RandomForestRegressor(random_state = 1,n_estimators = 100).fit(train_X,train_y)
predicted_y = reg.predict(test_X)
mean_absolute_error(test_y, predicted_y)

#output
164897.82257800733
{% endhighlight %}
<h5 class="fw-bold mt-5">
Conclusion
</h5>

The results improved slightly, but there’s still plenty of room for optimization. Techniques like feature selection, hyperparameter tuning, and smoothing target-encoded features can help push performance even further.

However, the goal of this walkthrough wasn’t to build the best model—it was to demonstrate how One-Hot Encoding and Target Encoding can be used to transform categorical data into a numerical form suitable for machine learning models.