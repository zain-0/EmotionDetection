from fastapi import FastAPI
from api.api import main_router

app = FastAPI(title="Emotions Detections") 

app.include_router(main_router)


@app.get('/')
def root():
    return {"hello": "world"}
