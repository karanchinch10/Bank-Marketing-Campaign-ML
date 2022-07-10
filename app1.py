import pandas as pd
import numpy as np
import sklearn 
import xgboost as xg
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle


pipe = pickle.load(open('model.pkl','rb'))
df = pd.read_csv("bank_data.csv")

st.markdown('<style>h1{color: white; background-color:darkblue;text-align: center;padding: 15px;font-size:200%}</style>', unsafe_allow_html=True)

st.title("Bank Marketing Campaign 🏦")
st.subheader("Predict Either New Customer will subscribed for term Deposit or Not??")
st.image("banking-marketing.jpg")
st.subheader("Enter the Customer Bank Details")

age = st.number_input("Enter the Age ",min_value=0,max_value=100,step=1)
job = st.selectbox("Job Type",('admin.', 'technician', 'services', 'management', 'retired', 'blue-collar',
 'unemployed', 'entrepreneur', 'housemaid', 'unknown', 'self-employed',
 'student'))
marital = st.selectbox("Marital",
('married', 'single', 'divorced'))
education = st.selectbox("Education Type",
('secondary', 'tertiary', 'primary', 'unknown'))
default = st.selectbox("Has credit in default?",('yes', 'no'))
balance = st.slider("Enter your Bank Balance(RS) ",min_value=0,max_value=100000,step=10)
housing = st.selectbox("Have Any Housing loan?",('yes', 'no'))
loan = st.selectbox("Have Any Personal loan?",('yes', 'no'))

st.subheader("Enter Info regarding Current Campaign")
contact = st.selectbox("Contact Type",
('unknown', 'cellular', 'telephone'))
day = st.slider("Enter Day ",min_value=0,max_value=35,step=1)
month = st.selectbox("Month",('may',
 'jun',
 'jul',
 'aug',
 'oct',
 'nov',
 'dec',
 'jan',
 'feb',
 'mar',
 'apr',
 'sep'))
duration = st.number_input("Call Duration (Sec)",min_value=0,max_value=10000,step=5)
campaign = st.slider("No of contact performed during this Compaign?",min_value=0,max_value=30,step=1)
previous = st.slider("No of contact performed in previous Compaign?",min_value=0,max_value=30,step=1)
poutcome = st.selectbox("Outcome of Previous Campaign",
('unknown', 'other', 'failure', 'success'))

st.write("Note: If customer is New for this campaign, poutcome will be unknown & previous contact=0")
st.write("")

#ordinal encoding 
dic_bin = {"yes":1,"no":0}
dic_contact={'unknown': 0, 'cellular': 1, 'telephone': 2}
dic_month={'sep': 0, 'apr': 1, 'mar': 2, 'feb': 3, 'jan': 4, 'dec': 5, 'nov': 6, 'oct': 7, 'aug': 8, 'jul': 9, 'jun': 10, 'may': 11}
dic_poutcome={'success': 0, 'unknown': 1, 'other': 2, 'failure': 3}


#Train model
X = df.drop('deposit',axis=1)
Y = df['deposit']

#StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=1)
for train_index,test_index in sss.split(X,Y):
    train_df = df.loc[train_index]
    test_df = df.loc[test_index]

#Train and Test dataset
X_train = train_df.drop("deposit",axis=1)
Y_train = train_df['deposit']

X_test = test_df.drop("deposit",axis=1)
Y_test = test_df['deposit']


#convert user input data into binary encode
default1 = dic_bin[default]
housing1 = dic_bin[housing]
loan1 = dic_bin[loan]
contact1 = dic_contact[contact]
month1 = dic_month[month]
poutcome1 = dic_poutcome[poutcome]

#Predict Complete New Test data
test_input = np.array([age,job,marital,education,default1, balance,housing1,loan1,contact1,day,month1,duration,campaign,previous,poutcome1],dtype=object).reshape(1,15)
#df4 = pd.DataFrame(test_input,columns=X_train.columns)

#predict result
res = pipe.predict(test_input)


result = st.button("Click to Predict")

if result:
    if int(res[0])==1:
        st.success("Will subscribed for deposit")
    elif int(res[0])==0:
        st.success("Will Not subscribed for deposit")
st.write("")

st.write("💝 Made By Karan Chinchpure 💝")


