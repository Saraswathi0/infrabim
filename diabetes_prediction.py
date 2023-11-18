

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split                   #modules import

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from urllib import request


r=pd.read_csv("dataset.csv")
      #dataset loading
r.head()

r.tail()

r['Outcome'].value_counts()

r.groupby('Outcome').mean()

r.describe()

x=r.drop(columns='Outcome',axis=1)
y=r['Outcome']

s=StandardScaler()
x= s.fit_transform(x)
     #data standardisation

y=r['Outcome']

X1, X2, y1, y2 = train_test_split(x, y, test_size=0.2, random_state=42) #data spliting

from sklearn import svm

classifier=svm.SVC(kernel='linear')            #training the svm classifier
classifier.fit(X1,y1)

m = LogisticRegression()
m.fit(X1, y1)
y_pred = m.predict(X1)
accuracy = accuracy_score(y1, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")                  #ACCURACY analysis of training data
report = classification_report(y1, y_pred)
print("Classification Report:\n", report)

y_pred = m.predict(X2)
accuracy = accuracy_score(y2, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")                  #ACCURACY analysis of testingdata

import streamlit as st

st.title('Diabetes Prediction App')

st.write('This app predicts the blood sugar level of a diabetes patient based on the provided features.')

st.header('Enter the features for prediction:')
with st.form(key='my_form'):
    X1 = st.number_input('Enter a number of preganices')
    X2 = st.number_input('Enter your glucouse level')
    X3 = st.number_input('Enter your blood pressure')
    X4 = st.number_input('Enter your skinthickness')
    X5 = st.number_input('Enter your insulin')
    X6 = st.number_input('Enter BMI')
    X7 = st.number_input('Enter DiabetesPedigreeFunction')
    X8 = st.number_input('Enter your age')
    submit = st.form_submit_button(label='Submit')
if submit:
   s1=[X1,X2,X3,X4,X5,X6,X7,X8]    #predicting the model
#changing data to numpy array
   array=np.asarray(s1)
#reshaping the array
   r1=array.reshape(1,-1)
   s2=s.transform(r1)
   print(s2)
   p1=classifier.predict(s2)
   print(p1)
   if(p1==1):
      st.write("diabetes")
   else:
      st.write("no diabetes")