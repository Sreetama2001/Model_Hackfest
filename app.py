from fastapi import FastAPI,Form,HTTPException
from pydantic import BaseModel
import uvicorn 
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# logic: int
# hackathon:int
# coding:int
# speaking:int
# eCourses:str
# selfLearn: str
# workshops:str
# certificates:str
# readingWriting:str
# memoryCapa:str
# subj:str
# career:str
# company:str
# seniors:str
# books:str
# management:str
# hardSmart:str
# workedteam:str
# introvert:str



@ app.post("/predict/")
async def receiveData(logic :int = Form(),hackathon:int =Form(),coding:int =Form(),speaking:int =Form(),eCourses:str =Form(), selfLearn: str =Form(),workshops: str =Form(),certificates:str =Form(),readingWriting:str =Form(),memoryCapa:str =Form(),subj:str =Form(),career:str =Form(),company:str =Form(),
seniors:str =Form(),books:str =Form(),management:str =Form(),hardSmart:str =Form(),workedteam:str =Form(),introvert:str =Form()):
    # print(logic)
    # print(hackathon)
    # print(coding)
    # print(speaking)
    # print(eCourses)
    # print(selfLearn)
    # print(workshops)
    # print(certificates)
    # print(readingWriting)
    # print(memoryCapa)
    # print(subj)
    # print(career)
    # print(company)
    # print(seniors)
    # print(books)
    # print(management)
    # print(hardSmart)
    # print(introvert)
    # print(workedteam)
    return {"logicical":logic,""}

    # return {"hello"}

if __name__ == '__main__':
    uvicorn.run(app,host="127.0.0.1",port=3000)

