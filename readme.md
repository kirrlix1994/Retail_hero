## The 2nd place solution of the [RetailHero Uplift Modelling contest ](https://retailhero.ai/c/uplift_modeling/overview)

### To open Notbebook for read, go to [nbviewer](https://nbviewer.jupyter.org/github/kirrlix1994/Retail_hero/blob/master/Retail_hero_contest_2nd_place_solution.ipynb)

## Problem

X5 has the ability to send SMS MESSAGES to customers in order to encourage them to make purchases. It is clear that it makes sense to make communication only for those customers who would not have made a purchase without it, and after it - will make a purchase. You need to develop an algorithm that can successfully predict which customers should send SMS messages and which should not.


## Evaluation

In uplift modeling tasks, clients from the test sample are ranked in descending order of communication efficiency. The top 30% (the most promising) are selected from the ranked list. The average added conversion rate is estimated for the selected 30%. In simple terms, the average increase in response is calculated when the client is affected.
