from sre_constants import SUCCESS
from fastapi import FastAPI,Form,HTTPException
from pydantic import BaseModel
import uvicorn 
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import pandas as pd

with open("weights\\decisiontree.pkl", 'rb') as f:
    model1=pickle.load(f)

with open("weights\\svm.pkl", 'rb') as f:
    model2=pickle.load(f)

with open("weights\\randomForest.pkl", 'rb') as f:
    model3=pickle.load(f)

with open("weights\\xboost.pkl", 'rb') as f:
    model4=pickle.load(f)

with open("weights\\oncampus.pkl", 'rb') as f:
    model5=pickle.load(f)   

def encoding_userInput(df):
    cols = df[["self-learning capability?", "Extra-courses did","Taken inputs from seniors or elders", "worked in teams ever?", "Introvert"]]
    for i in cols:
        cleanup_nums = {i: {"yes": 1, "no": 0}}
        df = df.replace(cleanup_nums)
    

    mycol = df[["reading and writing skills", "memory capability score"]]
    for i in mycol:
        cleanup_nums = {i: {"poor": 0, "medium": 1, "excellent": 2}}
        df = df.replace(cleanup_nums)


    # Label Encoding
    category_cols = df[['certifications', 'workshops', 'Interested subjects', 'interested career area ', 'Type of company want to settle in?', 
                        'Interested Type of Books']]
    for i in category_cols:
        df[i] = df[i].astype('category')
        df[i + "_code"] = df[i].cat.codes


    # Dummy Variable Encoding
    df = pd.get_dummies(df, columns=["Management or Technical", "hard/smart worker"], prefix=["A", "B"])
    print("the new df")
    print(df.head())
    userInput = []
    
    for i in range (1, df.shape[1]):
        userInput.append(df.iloc[0][i])
    # print(userInput)
    return(userInput)

app=FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080/",
    "http://localhost:8080/final/html/offcampus.html/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@ app.post("/predict/")
async def receiveData(logic :int = Form(),hackathon:int =Form(),coding:int =Form(),speaking:int =Form(),eCourses:str =Form(), selfLearn: str =Form(),workshops: str =Form(),certificates:str =Form(),readingWriting:str =Form(),memoryCapa:str =Form(),subj:str =Form(),career:str =Form(),company:str =Form(),
seniors:str =Form(),books:str =Form(),management:str =Form(),hardSmart:str =Form(),workedteam:str =Form(),introvert:str =Form()):

    dict_= {'Logical quotient rating':[logic], 'hackathons':[hackathon], 'coding skills rating':[coding],
       'public speaking points':[speaking], 'self-learning capability?':[selfLearn],
       'Extra-courses did':[eCourses], 'certifications':[certificates], 'workshops':[workshops],
       'reading and writing skills':[readingWriting], 'memory capability score':[memoryCapa],
       'Interested subjects':[subj], 'interested career area ':[career],
       'Type of company want to settle in?':[company],
       'Taken inputs from seniors or elders':[seniors],'Interested Type of Books':[books],
       'Management or Technical':[management], 'hard/smart worker':hardSmart, 'worked in teams ever?':[workedteam],
       'Introvert':[introvert]}

    df = pd.DataFrame.from_dict(dict_)
    userInput=encoding_userInput(df)

    print(userInput)
    # print(userInput.type)
    return {"SuccESS"}

if __name__ == '__main__':
    uvicorn.run(app,host="127.0.0.1",port=3000)

