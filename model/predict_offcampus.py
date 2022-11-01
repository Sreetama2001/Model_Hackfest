## predict 
import sys
import pickle
import numpy as np

with open("weights\\decisiontree.pkl", 'rb') as f:
    clf1=pickle.load(f)

with open("weights\\svm.pkl", 'rb') as f:
    clf2=pickle.load(f)

with open("weights\\randomForest.pkl", 'rb') as f:
    clf3=pickle.load(f)

with open("weights\\xboost.pkl", 'rb') as f:
    clf4=pickle.load(f)

# userdata =[[sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11], sys.argv[12], sys.argv[13], sys.argv[14], sys.argv[15], sys.argv[16], sys.argv[17], sys.argv[18], sys.argv[19],sys.argv[20],sys.argv[21] ]]

userdata = [[3, 7, 3, 1, 1, 'information security', 'game development', 0, 0, 'data engineering', 'Business process analyst', 'Finance', 1, 'Horror', 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]]

# Prediction By Decision Tree
print(clf1.predict(userdata)) 
classprobs1 = clf1.predict_proba(userdata)
predclassprob1 = np.max(classprobs1)

# Prediction By SVM
print(clf2.predict(userdata)) 
classprobs2 = clf2.decision_function(userdata)
predclassprob2 = np.max(classprobs2)

# Prediction By Random Forest
print(clf3.predict(userdata)) 
classprobs3 = clf3.predict_proba(userdata)
predclassprob3 = np.max(classprobs3)

# Prediction By XGBoost
print(clf4.predict(userdata)) 
classprobs4 = clf4.predict_proba(userdata)
predclassprob4 = np.max(classprobs4)

print(predclassprob1)
print(predclassprob2)
print(predclassprob3)
print(predclassprob4)