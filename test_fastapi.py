from fastapi import FastAPI

app = FastAPI()

@app.get("/")   #get, post, put, delete
def welcome():
    return {"message": "Hello FastAPI"}

@app.get("/home")
def welcome_home():
    return {"message": "Hello FastAPI Home"}

@app.post("/dummy")
def welcome_dummy(data):
    return {"message": data}


    