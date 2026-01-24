from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from ia import mon_ia

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restreins après déploiement
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    return {"reply": mon_ia(req.message)}

@app.get("/")
def root():
    return {"status": "NEXA backend online"}
