#  Bank Marketing | Who will Subscribed for deposit? 🏦
## Problem Statement:
The following project focus on the analysis of a dataset of **Bank Marketing** which contains data or information about customers and aims to get useful insights from the data and predict if a new customer will accept a deposit offer or not.
Also focus on to get important features which dominate most to define weather new customer will accept the offer or not?

![Semantic description of image](banking-marketing1.jpg "Image Title")
## Content:
This Dataset contains information of 10000+ bank customers data
## Dataset contains Total 17 columns/Features :

### Bank Client Data
* 1 - **age** : Age of customer(numeric)
* 2 - **job** : Job Type (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
* 3 - **marital** : marital status (categorical: 'divorced','married','single','unknown')
* 4 - **education** (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
* 5 - **default**: has credit in default? (categorical: yes/no)
* 6 - **balance**: bank balance(RS)
* 7 - **housing**: has housing loan? (categorical: yes/no)
* 8 - **loan**: has personal loan? (categorical: yes/no)

### Info related with the last contact of the current campaign:
* 9 - **contact**: contact communication type (categorical: 'cellular','telephone')
* 10 - **month**: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
* 11 - **day**: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
* 12 - **duration**: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

### Other Attributes:
* 13 - **campaign**: number of contacts performed during this campaign and for this client (numeric, includes last contact)
* 14 - **pdays**: number of days that passed by after the client was last contacted from a previous campaign (numeric;-1 means client was not previously contacted)
* 15 - **previous**: number of contacts performed before this campaign and for this client (numeric)
* 16 - **poutcome**: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

### Output variable (desired target):
* 17 - **deposit** - has the client subscribed a term deposit? (binary: yes/no)
## My Work:
- I have made this model which will predict either new customer will accept a deposit offer or not
- I have done stepwise EDA (Exploratory Data Analysis) then visualizatiion of categ and Num features to get some idea about important features or correlation
- Then I have done Feature Engineering which inclueds detecting and handling of outliers and features extraction based on my domian knowledge and visualization followed by label encoding and scaling 
- I have train ML models with multiples algorithms on same data in order to Analysed & compare performance of differents models based of accuracy and complexity
- After comparing I got well accuracy with 2 algorithms which are Random Forest and Xg boost with accuracy 85% and Recall Score 88% 
- Technical tools or library used --Python,numpy,pandas,sklearn,matplotllib,xgboost 
- 
  -  <a href="https://www.kaggle.com/code/karanchinchpure/bank-marketing-who-will-subscribe-for-deposit">View on Kaggle</a> 💝
  -  👉<a href="#">View On Github</a> 💝
