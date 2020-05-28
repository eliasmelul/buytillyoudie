# Buy 'Till You Die
This repository introduces the Kaplan-Meier and BG/NBD models, as well as logistic regression with endogeneity correction using Inverse Mills Ratio, to maximize profitability of a company within the following CRM framework:

<img src="https://i.ibb.co/3TZKD5v/Framework.png" alt="Framework" border="0">

In this repository, we will use <a href="https://www.kaggle.com/blastchar/telco-customer-churn"> Telco's Customer Churn</a> dataset as an ongoing example. 

Happy Reading!

---
___

<a href='https://github.com/eliasmelul/'> <img src='https://s3.us-east-2.amazonaws.com/wordontheamazon.com/NoMargin_NewLogo.png' style='width: 15em;' align='right' /></a>
# Buy 'Till You Die
### Predicting Customer Churn
___
<h4 align="right">by Elias Melul, Data Scientist </h4> 

___
## Marketing Theory
<img src="https://i.ibb.co/GHTPdbc/jeffquote1.jpg" alt="jeffquote1" border="0" width = "500">

Customer centricity is essential for a successful strategy. Without the customer, there are no sales, no revenue, no profit... no company! That's why customer relationship management has become a critical aspect of any company. So what is customer relationship management?

Customer Relationship Management is the process of aligning products and services that a firm offers to an understanding of the customer's needs and goals, use cases, and priorities. Hence, companies should not seek for the right customer for their product, but rather the right product for the customer. That is why in this notebook we will approach CRM from a resonance focus - that is, shift product management or marking initiatives to fit customer's needs by differentiating on the most significant elements to customers.

Throughout this notebook, we will make reference to the following framework, under the assumption that marketing initiatives try to maximize profit. Fair assumption, right?

<img src="https://i.ibb.co/3TZKD5v/Framework.png" alt="Framework" border="0">

We also need to understand some basic marketing concepts before we begin. More specifically, we will look at Sunil Gupta's definition of <a href="https://www.researchgate.net/publication/237287176_Modeling_Customer_Lifetime_Value">Customer Lifetime Value</a>, CLV for short, and the Customer Equity equation which is based on the framework proposed.

### Customer Lifetime Value and Customer Equity
CLV is typically defined as the present value of all future profits that a customer generates over his/her life with the firm. 

<img src="https://i.ibb.co/0V1FRn4/PVFormula.png" alt="PVFormula" border="0">

While similar to the idea of discounted cash flows used in Finance, the CLV is calculated on individual customers or customer segments, and the CLV incorporates the posibility that a customer may leave (churn) to consume the competitor's product or service. Researchers and practitioners have developed numerous ways of computing the CLV, all valid but with different advantages and dissadvantages. For instance:

<img src="https://i.ibb.co/HgMbP8j/CLVComp.png" alt="CLVComp" border="0">

Gupta and Lehmann (2003, 2005) showed that if margins (p-c) and retention rates are constant over time, using an infinite horizon, one can simplify the CLV through an infinite geometric series to the following:

<img src="https://i.ibb.co/6mzqQw6/CLVGupta.png" alt="CLVGupta" border="0">

In other words, the CLV simplifies to be the profit margin times the margin multiple! So what does the margin mutiple mean? It is simply the amount of times the firm will see a customer generate profits during his/her relationship with the firm.

This metric is key to quantifying the revenues and benefits of acquiring new customers or retaining them.

<img src="https://i.ibb.co/p2mmtHp/Color-Framework.png" alt="Color-Framework" border="0">

The above equation shows the expected customer equity based on the variables in our framework. From here on out, everything that we try to do is intended to increase customer equity, which in turn increases profitability. Mostly, we will deal with customer acquisition and retention. We will start with the latter.

____
