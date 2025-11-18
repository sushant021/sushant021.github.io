---
layout: post
title:  "Why Is It Necessary To Balance An Imbalanced Dataset"
date:   2025-11-12 14:14:11 -0700
tags: [data-science, python] 
---

An imbalanced dataset is kind of a dataset where the distribution of classes is highly uneven. For example in a credit card transactions data, you may see a few fraud transactions out of 1000s of transactions. In this case, if you classify data as "legit" and "fraud", the data itself is highly skewed towards "legit" class. This is what "imbalanced" means. Imbalanced datasets are more common than we think. For example, an insurance fraud dataset, a defective product dataset in a manufacturing plant, click-through-rate in an ad dataset etc. These are all examples, where if you classify the data as "true" and "false", the whole data skews towards one side. This imbalance can also occur in a multi-class classification dataset, where some classes have far fewer samples than others.

Such datasets have to be balanced before we train a model on them. But why ? What's wrong if we just train a model with the imbalanced data as it is ? That's what we'll see in this post.  

What we'll do is we'll train a logistic regression model first on the imbalanced data and see the results. And then we'll do the same with a balanced data and then compare the results.

We'll use the Credit Card Fraud Detection Dataset from Kaggle. You can get the data and learn about it here: <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud">https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud</a>

<h5 class="fw-bold mt-5"> Let's Not Balance </h5>

Let's work with the data as it is for now. Let's not balance. 

Let's import the libraries. 

{% highlight python %}

import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

{% endhighlight %}

Let's load the data.

{% highlight python %}

df =pd.read_csv("../input/creditcardfraud/creditcard.csv")
df.head() #if you want to check if it loaded correctly 
{% endhighlight %}

{% highlight python %}

#Output

Time	V1	V2	V3	V4	V5	V6	V7	V8	V9	...	V21	V22	V23	V24	V25	V26	V27	V28	Amount	Class
0	0.0	-1.359807	-0.072781	2.536347	1.378155	-0.338321	0.462388	0.239599	0.098698	0.363787	...	-0.018307	0.277838	-0.110474	0.066928	0.128539	-0.189115	0.133558	-0.021053	149.62	0
1	0.0	1.191857	0.266151	0.166480	0.448154	0.060018	-0.082361	-0.078803	0.085102	-0.255425	...	-0.225775	-0.638672	0.101288	-0.339846	0.167170	0.125895	-0.008983	0.014724	2.69	0
2	1.0	-1.358354	-1.340163	1.773209	0.379780	-0.503198	1.800499	0.791461	0.247676	-1.514654	...	0.247998	0.771679	0.909412	-0.689281	-0.327642	-0.139097	-0.055353	-0.059752	378.66	0
3	1.0	-0.966272	-0.185226	1.792993	-0.863291	-0.010309	1.247203	0.237609	0.377436	-1.387024	...	-0.108300	0.005274	-0.190321	-1.175575	0.647376	-0.221929	0.062723	0.061458	123.50	0
4	2.0	-1.158233	0.877737	1.548718	0.403034	-0.407193	0.095921	0.592941	-0.270533	0.817739	...	-0.009431	0.798278	-0.137458	0.141267	-0.206010	0.502292	0.219422	0.215153	69.99	0
5 rows × 31 columns

{% endhighlight %}

The very last column here shows the class - 0 or 1. 

Let's check how imbalanced the data is. 

{% highlight python %}
data_count = df['Class'].value_counts()
print(data_count)

#Output
Class
0    284315
1       492
Name: count, dtype: int64
{% endhighlight %}

Here we see there are 492 rows that are under class "1", which means they are classified as fraud transactions. The 284315 rows are under class "0" meaning they are legit transactions.  
We can see that the data is highly imbalanced. Out of whole data, only 492 (0.1730%) are fraud transactions. This is what we call a minority class. The legit transactions are the majority class.

Let's train our logistic regression model with this data just as it is. 
Let's split the training / test, fit the model and check the accuracy. 

{% highlight python %}

inputs = df.drop("Class", axis="columns")
targets = df.Class
x_train, x_test, y_train, y_test = train_test_split(
    inputs, targets, test_size=0.2, random_state=10)
clf = LogisticRegression(max_iter = 10000).fit(x_train,y_train)
y_predicted = clf.predict(x_test)
accuracy = clf.score(x_test, y_test)
print(accuracy)

#Output
0.9988588883817282

{% endhighlight %}

We got a 99.88% accuracy without doing any balancing. And that's a really good result, why is it necessary to balance then, you might be thinking. Well, that'll be answered by this confusion matrix. 

{% highlight python %}
cm = metrics.confusion_matrix(y_test, y_predicted)

#let's make it a bit easier to read with a plot of this matrix.  
sns.heatmap(cm, annot=True, cmap='Blues_r' ,fmt='d')
plt.ylabel("Actual Values")
plt.xlabel("Predicted Values")
plt.show()

{% endhighlight %}

![First Confusion Matrix](/assets/images/results_1.png)

Here we see that 36 of fraud transactions (out of 94) have been classified as legit (False Negative). But only 29 of legit transactions (out of 56868) have been classified as fraud(False positive). Or in other words,only 58 fraud transactions(out of 94, which makes it 61.7%) were accurately classified as fraud. But 56839 legit transactions (out of 56868, which makes it 99.94%) were accurately classified as legit.

Which means that our model is doing an excellent job of predicting the legit transactions but very poor predicting the fraud transactions. Which means that 99.8% accuracy is pretty much the accuracy to predict legit transactions, not fraud. That accuracy is mostly just the model's ability to predict legit transactions. But that's not what we are trying to achieve, are we ? We want the model to predict the fraud transactions with better accuracy. 

<h5 class="fw-bold mt-5">Balancing with Undersampling</h5>

We will balance the data by undersampling the legit transactions in the training dataset. We will not balance the test data. The imbalance must be preserved in the test set because the real world data is imbalanced. It doesn't make sense to test on the balanced data if we are eventually going to use the model for real world imbalanced data.

First, we will recreate the training data from x_train and y_train to separate legit and fraud transactions.

{% highlight python %}
train_data = pd.concat([x_train, y_train], axis=1)

#Separate legit and fraud transactions.
legit_train = train_data[train_data.Class == 0]
fraud_train = train_data[train_data.Class == 1]

#Check how many frauds we got so that we can undersample legit transactions to that number.
print(legit_train.shape, fraud_train.shape)

#Output
(227447, 31) (398, 31)

{% endhighlight %}

<h5 class="fw-bold mt-5">Random Undersampling </h5>

So, we see that there are 227447 legit transactions and 398 fraud transactions in our training data set. We will balance this dataset by randomly selecting 398 legit transactions out of 227447. This is called Random Undersampling.

{% highlight python %}

legit_train = legit_train.sample(n=398)
legit_train.shape

#output
(398, 31)

{% endhighlight %}

Now that we have same number of fraud and legit transactions, let's join them back to recreate a complete balanced training dataset.

{% highlight python %}

train_data = pd.concat([legit_train, fraud_train], axis=0)
train_data.shape

#output
(796, 31)

{% endhighlight %}

Now, let's separate features and targets from the new training dataset. 

{% highlight python %}

x_train_new = train_data.drop('Class', axis='columns')
y_train_new = train_data.Class

{% endhighlight %}

We don't have to bother with the test dataset because it will stay the same. So, we will just fit the model with new x_train_new, y_train_new and old x_test, y_test.

{% highlight python %}

clf = LogisticRegression(max_iter=10000).fit(x_train_new, y_train_new)
y_predicted = clf.predict(x_test)
accuracy = clf.score(x_test, y_test)
print(accuracy)

#output
0.9539342017485342

{% endhighlight %}

We got an accuracy of 95.39%. Not bad. Let's check the confusion matrix.

{% highlight python %}
cm = metrics.confusion_matrix(y_test, y_predicted)
sns.heatmap(cm, annot=True, cmap='Blues_r', fmt='d')
plt.ylabel("Actual Values")
plt.xlabel("Predicted Values")
plt.show()

{% endhighlight %}

![First Confusion Matrix](/assets/images/results_2.png)

From the confusion matrix, we see that 85 fraud transactions (out of 94, which makes it 90.4% ) were accurately classified as fraud. And 54781 legit transactions (out of 56868, which makes it 96.3%) were accurately classified as legit. Only 9 fraud transactions were classified as legit transactions. This result shows that even though we got less accuracy, this model is better at predicting the fraud transactions.

<h5 class="fw-bold mt-5"> Conclusion </h5>
In real world, most of the data are always imbalanced. For example, google ads click through rate, faulty products in an assembly line, spam emails etc. If we don't do anything in such imbalanced dataset, the model tends to be biased towards majority class, whatever that is. And that is a problem because we are trying to predict the minority class, not the majority. So, the biasness of the model towards majority class must be addressed. We do this by undersampling the majority class (decreasing the size of majority class) or oversampling the minority class (increasing the size of minority class).

From above results, we see that even though we get high accuracy without balancing, that accuracy shows how great it is at predicting legit transactions and how bad it is at predicting fraud transactions. But we always need to predict the fraud transactions or spams or faulty products which are always a minority in these kinds of dataset.

And that is why balancing is necessary.