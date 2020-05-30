___

<a href='https://github.com/eliasmelul/'> <img src='https://s3.us-east-2.amazonaws.com/wordontheamazon.com/NoMargin_NewLogo.png' style='width: 15em;' align='right' /></a>
# Buy 'Till You Die
### Predicting Customer Churn
___
<h4 align="right">by Elias Melul, Data Scientist </h4> 

___
This repository introduces the Kaplan-Meier and BG/NBD models, as well as logistic regression with endogeneity correction using Inverse Mills Ratio, to maximize profitability of a company within a CRM framework.
In this repository, we will use <a href="https://www.kaggle.com/blastchar/telco-customer-churn"> Telco's Customer Churn</a> dataset as an ongoing example. 

Happy Reading!

---
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

What are the assumptions that this Infinite Horizon CLV makes?
1. Customers have a constant profit margin, m, over time
2. Customers have a constant retention rate, alpha, over time
3. Discount rate is constant over time
4. Value is estimated over an infinite horizon

This metric is key to quantifying the revenues and benefits of acquiring new customers or retaining them.

<img src="https://i.ibb.co/p2mmtHp/Color-Framework.png" alt="Color-Framework" border="0">

The above equation shows the expected customer equity based on the variables in our framework. From here on out, everything that we try to do is intended to increase customer equity, which in turn increases profitability. Mostly, we will deal with customer acquisition and retention. We will start with the latter.

```
def margin_multiplier(alpha, discountrate):
    M = alpha/(1+discountrate-alpha)
    return M
def clvCalc(margin, multiplier):
    clv = margin*multiplier
    return clv
```
Let's assume that the retention rate is 90%, and that the discount rate is 10%. What is the margin multiplier?

If we assume that the margin is $60, what is the CLV?
```
print(f"The margin multiple is {round(margin_multiplier(0.9,0.1),2)}.")
print(f"The CLV is ${round(clvCalc(60,margin_multiplier(0.9,0.1)))}.")
```
The margin multiple is 4.5.
The CLV is $270.

But what if the firm experiences growth over time?

Simple. Gupta and Lehmann have shown that when margins grow at a constant rate g, the CLV becomes:

<img src="https://i.ibb.co/JjF3hNB/CLVgrowth.png" alt="CLVgrowth" border="0">

So let's assume that the company we were talking about experiences 3% growth. How would our evaluation change?

```
def margin_multiplier(alpha, discountrate, growth = None):
    if growth == None:
        M = alpha/(1+discountrate-alpha)
    else:
        M = alpha/(1+discountrate-alpha*(1+growth))
    return M

print(f"The margin multiple is {round(margin_multiplier(0.9,0.1,0.03),2)}.")
print(f"The CLV is ${round(clvCalc(60,margin_multiplier(0.9,0.1,0.03)))}.")
```
The margin multiple is 5.2.
The CLV is $310.

Great stuff! Now, let's introduce our data.

## Data Understanding

Throughout the notebook, we will use <a href="https://www.kaggle.com/blastchar/telco-customer-churn">Telco Customer Churn</a> data. It is a sample dataset that include one row per customer including the following features and characteristics of a customer:
* customerID
* gender
* SeniorCitizen - whether a customer is a senior citizen or not (binary)
* Partner - whether the customer is single by legal status or not (Yes or No)
* Dependents - whether the customer has legal dependents of not (Yes or No)
* tenure - the number of months the customer has been with the company
* PhoneService - whether the customer's account includes phone service (Yes or No)
* MultipleLines - whether the customer's account includes multiple lines or just one (Yes, No, or No phone service)
* InternetService - whether the customer's account includes internet serice (DSL, Fiber optic, or No)
* OnlineSecurity - whether the customer's account includes online security service (Yes, No, or No internet service)
* OnlineBackup - whether the customer's account includes online backup service (Yes, No, or No internet service)
* DeviceProtection - whether the customer's account includes protection service (Yes, No, or No internet service)
* TechSupport - whether the customer's account includes IT support (Yes, No, or No internet service)
* StreamingTV - whether the customer's account includes streaming services (Yes, No, or No internet service)
* StreamingMovies - whether the customer's account includes streaming services (Yes, No, or No internet service)
* Contract - the type of contract the customer has with Telco (Month-to-month, One year, or Two year)
* PaperlessBilling - whether the customer signed up to paperless billing (Yes or No)
* PaymentMethod - method of payment of contract (Electronic check, Mailed check, Bank transfer (automatic), or Credit card (automatic)
* MonthlyCharges - the monthly charges (in USD) of each customer
* TotalCharges - the total charges (in USD) Telco has processed for each customer during their lifetime
* Churn - whether the customer has churned or not (Yes or No)

Note that Telco is not an actual company, but rather sample data published by <a href="https://www.ibm.com/support/knowledgecenter/SSEP7J_11.1.0/com.ibm.swg.ba.cognos.ig_smples.doc/c_inst_installc8samples.html">IBM</a>.

```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

data = pd.read_csv("C:/Users/melul/Desktop/BTYD/WA_Fn-UseC_-Telco-Customer-Churn.csv", index_col=0)

print(data.shape)
data.iloc[0] #Preferable data.head() but this is enough for github readme
```
(7043, 20)
|Name              |First Value|
| -----------------|-----------|
|gender            |    Female|
|SeniorCitizen     |    0|
|Partner  |       Yes|
|Dependents    |           No|
|tenure       |      1|
|PhoneService |         No|
|MultipleLines  |  No phone service|
|InternetService   |   DSL|
|OnlineSecurity   |     No|
|OnlineBackup   |     Yes|
|DeviceProtection   |    No|
|TechSupport   |          No|
|StreamingTV    |     No|
|StreamingMovies     |     No|
|Contract   |  Month-to-month|
|PaperlessBilling  |     Yes|
|PaymentMethod  |  Electronic check|
|MonthlyCharges  |   29.85|
|TotalCharges  |   29.85|
|Churn   |  No|


