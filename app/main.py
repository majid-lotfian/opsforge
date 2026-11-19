from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .agent import analyze
from .config import settings

app = FastAPI(title="OpsForge", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[s.strip() for s in settings.cors_origins.split(",")],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

class Incident(BaseModel):
    title: str = Field(min_length=5, max_length=300)
    description: str = Field(min_length=10, max_length=10000)

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/analyze")
def analyze_incident(incident: Incident):
    try:
        return analyze(incident.title, incident.description)
    except Exception as exc:
        # Never leak Azure/Databricks credentials or raw connector exceptions to callers.
        raise HTTPException(status_code=502, detail="Incident analysis failed; inspect server logs") from exc

@app.get("/")
def index():
    return FileResponse(Path(__file__).with_name("index.html"))
