from fastapi import FastAPI
from pymongo import MongoClient
import os

app = FastAPI()

# Atlas 接続
client = MongoClient(os.getenv("MONGO_URI"), tls=True)
db = client[os.getenv("MONGO_DB")]

@app.get("/")
def root():
    return {"message": "FastAPI Render Server is running"}

@app.get("/ping")
def ping():
    try:
        db.command("ping")
        return {"status": "ok", "message": "Connected to Atlas"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
