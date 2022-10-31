import sys
import pickle
import numpy as np

with open("weights\\oncampus.pkl", 'rb') as f:
    clf1=pickle.load(f)

userdata=[[2, 1, 6, 1,1,1]]
                

# print(clf1.predict(userdata)) 
res={1:"Congratulations, You will surely get placed!", 0: " Based upon the previous years data you may not get placed. But keep working hard!"}

# for i in clf1.predict(userdata):
#     if i == 1:
#         print(res[i])

print(res[clf1.predict(userdata)[0]]) 