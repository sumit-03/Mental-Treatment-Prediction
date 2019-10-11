# This is the soluntion for Online hackathon

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the csv file and making dataframe object
input_data=pd.read_csv("C:\\Users\\DELL\\Desktop\\devenger\\trainms.csv",skiprows=0)

# test_set for which i have to make prediction
test_data = pd.read_csv("C:\\Users\\DELL\\Desktop\\devenger\\testms.csv",skiprows=0)

# printing columns present in dataset
col = []
for i in input_data.columns:
    col.append(i)
print(col)

# columns to be dropped which has less affect on predicton
columns_to_drop = ['s.no',
 'Timestamp',
 'Age',
 'Gender',
 'Country',
 'state',
 'no_employees',
 'tech_company',
 'seek_help',
 'anonymity',
 'leave',
 'mental_health_consequence',
 'phys_health_consequence',
 'coworkers',
 'supervisor',
 'mental_health_interview',
 'phys_health_interview',
 'mental_vs_physical',
 'obs_consequence',
 'comments']

# Dropping the columns from train_set and test_set
input_data = input_data.drop(columns_to_drop, axis = 1)
test_data = test_data.drop(columns_to_drop, axis = 1)

# Perfroming data pre-proceesing
input_data["work_interfere"].fillna(method = 'backfill', inplace = True)
test_data["work_interfere"].fillna(method = 'backfill', inplace = True)
input_data.dropna(inplace = True)
test_data.dropna(inplace = True)

# checking the count of null value in train_set and test_set
input_data.isnull().sum()
test_data.isnull().sum()

# Replacing hard coded string to simple 'No'
input_data.replace({"benefits":"Don't know", "care_options":"Not sure", "wellness_program":"Don't know"}, value = "No",inplace = True)
test_data.replace({"benefits":"Don't know", "care_options":"Not sure", "wellness_program":"Don't know"}, value = "No",inplace = True)

# converting dataset into numeric values using encoder
from sklearn.preprocessing import LabelEncoder
l_enc=LabelEncoder()
input_data = input_data.apply(l_enc.fit_transform)
test_data = test_data.apply(l_enc.fit_transform)

# separatig deoendent and independent variables
train_y=input_data.iloc[:,2].values
input_data=input_data.drop("treatment",axis=1)
train_x = input_data.iloc[:,:].values

# separting independent variables
test = test_data.iloc[:,:].values

#from sklearn.tree import DecisionTreeClassifier
#clfr = DecisionTreeClassifier()

#from sklearn.naive_bayes import GaussianNB
#clfr=GaussianNB()

from sklearn.ensemble import RandomForestClassifier
clfr=RandomForestClassifier()

#from sklearn.svm import SVC
#clfr=SVC(gamma='auto')

#from sklearn.naive_bayes import MultinomialNB
#clfr= MultinomialNB()

# fitting the model and prediction
clfr.fit(train_x, train_y)
y_pred = clfr.predict(test)

# making index column which represents row numbers
sl_no = np.arange(1, 260)

# converting from 0/1 to yes/no
y_pred = l_enc.inverse_transform(y_pred)

# finally making csv file for subimission
pred1 = pd.DataFrame({'s.no':sl_no, 'treatment':y_pred}).to_csv("data_MNB.csv", index=False)

# I got 78% accuracy while submitting this solution at online Hackathon








