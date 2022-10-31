import numpy as np 
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from catboost import CatBoostClassifier
import pickle

df = pd.read_csv("data\on_campus.csv")

df = df.drop(['Age'],axis=1)


# print(df.head())
# Number Encoding
# col= df[["Gender"]]
# for i in col:
#     cleanup_nums = {i: {"Female": 1, "Male": 0}}
#     df = df.replace(cleanup_nums)

# col= df[["Stream"]]
# for i in col:
#     cleanup_nums = {i: {"Electronics And Communication": 1,"Computer Science": 2,"Information Technology":3,"Mechanical":4}}
#     df = df.replace(cleanup_nums)

# print(df.head()) 

'''
electronics And communication --> 3
Computer Science --> 1
Information Technology--> 4
 Mechanical --> 5
 Civil --> 2

male --> 1
female --> 0

'''
pre = preprocessing.LabelEncoder()

df["Gender"] = pre.fit_transform(df["Gender"])
df["Stream"] = pre.fit_transform(df["Stream"])

# print(df.head())

X = df[['Gender', 'Internships', 'CGPA', 'Hostel',
       'HistoryOfBacklogs', 'Stream']]

y = df["PlacedOrNot"]

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=100)

clf = CatBoostClassifier(
    
    iterations = 5, 
    learning_rate = 0.1, 
    loss_function='CrossEntropy',
    
).fit(x_train, y_train)


pred = clf.predict(x_test)
acc = accuracy_score(y_test, pred)
print ("Accuracy is ",acc)

with open("oncampus.pkl", "wb") as f:
    pickle.dump(clf, f)