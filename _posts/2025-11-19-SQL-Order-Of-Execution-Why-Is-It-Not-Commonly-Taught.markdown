---
layout: post
title:  "SQL Order of Execution - A Topic Seldom Taught"
date:   2025-11-18 15:14:11 -0700
tags: [sql, data-analysis] 
---


Through my entire Bachelor's degree and throughout my other training
programs, nobody ever taught the SQL Order of Execution. Everybody kept
focusing on clauses, joins and just the basic process of writing a
query. It could be because it may be a slightly advanced topic but it is
super important nonetheless. Well, maybe like most topics in the
programming world, it's probably something many of us have to
self-learn. Still, it's something I find crucial as someone who works
with data, and I hope this helps someone out there.


<h5 class="mb-4 fw-bold"> Why Order of Execution Matters </h5>

Most people learn SQL the way I did: *SELECT this, FROM that, WHERE something*. And for a while, that works. Until it doesn’t.

Once you start writing more complex queries—especially those involving aggregations, window functions, nested subqueries, or tricky filtering logic—you begin to hit strange errors or unexpected results. You start to wonder:

“Why can’t I use this alias in the WHERE clause?”

“Why does this GROUP BY behave weirdly?”

“Why do some filters happen before others?”

“Why do my window functions run last even though they appear early in the query?”

The answer to all of these is the same: **SQL does not execute in the order you write it.**

Understanding SQL’s actual order of execution changes everything. It makes your queries more predictable, helps you debug faster, and transforms SQL from a memorized set of rules into a logical, understandable process.


<h5 class="mb-4 fw-bold"> The Actual SQL Order of Execution </h5>

Here’s the real order SQL follows behind the scenes (simplified but accurate enough for 99% of use cases):

1. FROM / JOIN
2. WHERE
3. GROUP BY
4. HAVING
5. SELECT (including expressions and column aliases)
6. Window Functions
7. DISTINCT
8. ORDER BY
9. LIMIT / OFFSET


<h5 class="mb-4 fw-bold"> Example: Why You Can’t Use Aliases in the WHERE Clause </h5> 

Consider the following query:

``` sql
SELECT price * quantity AS total_revenue
FROM sales
WHERE total_revenue > 1000;
```

This will throw an error in most SQL engines. Why?

Because SQL executes the WHERE clause before the SELECT clause.
At the time SQL processes WHERE, the alias `total_revenue` does not exist yet.

The Correct Way:
``` sql
SELECT price * quantity AS total_revenue
FROM sales
WHERE price * quantity > 1000;
```

<h5 class="mb-4 fw-bold"> Example: GROUP BY Happens Before SELECT </h5>
Let's take a look at this another classic beginner confusion. 

``` sql
SELECT customer_id, SUM(amount) AS total_spent
FROM transactions
WHERE total_spent > 500;   
GROUP BY customer_id;
```
Again, SQL evaluates in this order:

FROM >
WHERE >
GROUP BY >
SELECT

So `total_spent` doesn’t exist when WHERE runs.

The Correct Way — Use HAVING

``` sql
SELECT customer_id, SUM(amount) AS total_spent
FROM transactions
GROUP BY customer_id
HAVING SUM(amount) > 500;
```
This is a perfect example of why order of execution matters—many SQL learners simply memorize “HAVING is for aggregates,”. Now you know why and knowing why helps you understand and write better queries with it. 


<h5 class="mb-4 fw-bold"> Example: Window Functions Come Late </h5>

Window functions are powerful, but they obey strict rules because they happen near the end of execution.

Take the following example.
``` sql
SELECT
    customer_id,
    amount,
    ROW_NUMBER() OVER (ORDER BY amount DESC) AS rank
FROM transactions
WHERE rank = 1;  
```
This will not work. Why? Because `rank` is created late, after WHERE, so the WHERE clause has no idea it exists.

The Correct Way - Use Subquery or Common Table Expression (CTE)

``` sql
WITH ranked AS (
    SELECT 
        customer_id,
        amount,
        ROW_NUMBER() OVER (ORDER BY amount DESC) AS rank
    FROM transactions
)
SELECT *
FROM ranked
WHERE rank = 1;
```
Window functions are evaluated *after* SELECT but *before* ORDER BY, so they need their own step if you want to filter on them.

<h5 class="mb-4 fw-bold"> A More Realistic Example</h5>

Here’s a more realistic example that we may encounter in actual work. 

Question:

*Find each customer’s largest purchase, then return only the top 5 customers by purchase size.*

``` sql
WITH ranked_purchases AS (
    SELECT 
        customer_id,
        amount,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY amount DESC) AS purchase_rank
    FROM transactions
)
SELECT customer_id, amount
FROM ranked_purchases
WHERE purchase_rank = 1
ORDER BY amount DESC
LIMIT 5;
```
Let's break this down step by step. 

1. Start with outer query - FROM

    FROM `ranked_purchases` executes first. Upon seeing `ranked_purchases`, it then executes the CTE:
        
    ``` sql
    SELECT 
        customer_id,
        amount,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY amount DESC) AS purchase_rank
    FROM transactions
    ```
    This CTE has its own internal execution order :
    1. FROM Transactions
    2. Window Function ROW_NUMBER() - SQL partitions rows by customer_id, orders the rows by amount DESC, and assigns row numbers to them, now named "purchase_rank".  
    3. SELECT the columns customer_id, amount, purchase_rank                
<br>


2. Back to outer query - WHERE

    Now SQL treats the CTE result as a table `ranked_purchases` that has columns: customer_id, amount, purchase_rank. 
    Filtering by `purchase_rank = 1` gives the top row for each customer, i.e. each customer's highest purchase. 

3. SELECT customer_id, amount - select the required columns

4. ORDER BY - order by DESC

5. LIMIT - limit only top 5 rows


Once you understand the order of execution, queries like this become intuitive instead of trial-and-error.

<h5 class="mb-4 fw-bold">Conclusion</h5>

If you're new to SQL, learning the order of execution will save you hours of senseless debugging, make JOINs, GROUP BYs and HAVINGs more logical and meaningful, clarify when and why CTEs or subqueries are required. Overall it just helps you write cleaner queries and understand complex queries better. 

It’s one of those topics that nobody teaches you early, yet it unlocks a massive shift in how you think about SQL. If you’re just starting out, don’t feel bad—every seasoned data person has had that moment of realizing “SQL doesn’t run top to bottom.”

Now you’re ahead of the curve.
