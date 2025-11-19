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


<h5 class="fw-bold mt-5">
Why Order of Execution Matters
</h5>

Most people learn SQL the way I did: SELECT this, FROM that, WHERE something. And for a while, that works. Until it doesn’t.

Once you start writing more complex queries—especially those involving aggregations, window functions, nested subqueries, or tricky filtering logic—you begin to hit strange errors or unexpected results. You start to wonder:

“Why can’t I use this alias in the WHERE clause?”

“Why does this GROUP BY behave weirdly?”

“Why do some filters happen before others?”

“Why do my window functions run last even though they appear early in the query?”

The answer to all of these is the same: **SQL does not execute in the order you write it.**

Understanding SQL’s actual order of execution changes everything. It makes your queries more predictable, helps you debug faster, and transforms SQL from a memorized set of rules into a logical, understandable process.


<h5 class="fw-bold mt-5">
The Actual SQL Order of Execution (simplified but accurate enough for 99% of use cases):
</h5>
1.  FROM
2.  JOIN
3.  ON
4.  WHERE
5.  GROUP BY
6.  HAVING
7.  SELECT
8.  DISTINCT
9.  WINDOW FUNCTIONS
10. ORDER BY
11. LIMIT / OFFSET

This means that even though SELECT comes first in how you write a query, it’s one of the last things the database evaluates.

Let’s walk through this using a simple example.


<h5 class="fw-bold mt-5">
Example: Why Aliases Can't Be Used in WHERE
</h5>
``` sql
SELECT price * quantity AS total_revenue
FROM sales
WHERE price * quantity > 1000;
```
This will throw an error in most SQL engines. Why?

Because SQL executes the WHERE clause before the SELECT clause.
At the time SQL processes WHERE, the alias total_revenue does not exist yet.

The correct way:

``` sql
SELECT price * quantity AS total_revenue
FROM sales
WHERE price * quantity > 1000;
```


<h5 class="fw-bold mt-5">
Example: GROUP BY Comes Before SELECT
</h5>
``` sql
SELECT customer_id, SUM(amount) AS total_spent
FROM transactions
GROUP BY customer_id
HAVING SUM(amount) > 500;
```

The HAVING clause works because SQL processes GROUP BY before HAVING but
after WHERE.


<h5 class="fw-bold mt-5">
Example: Window Functions Come Late
</h5>
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

Window functions run after SELECT but before ORDER BY, so filtering on
them requires a subquery or CTE.


<h5 class="fw-bold mt-5">
Final Thoughts
</h5>
Understanding SQL's true order of execution turns SQL from a
memorization exercise into a logical workflow. It helps you debug
faster, write cleaner queries, and fully leverage SQL's
power---especially as you move into analytics, engineering, or data
science.





