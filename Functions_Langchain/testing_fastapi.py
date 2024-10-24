from fastapi import FastAPI
import uvicorn

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

if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=80)


# uvicorn Functions_Langchain.testing_fastapi:app --reload    