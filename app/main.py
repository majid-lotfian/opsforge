from pathlib import Path
import logging
import uuid
import time

log = logging.getLogger('uvicorn.error')
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
    request_id = uuid.uuid4().hex
    started = time.monotonic()
    try:
        result = analyze(incident.title, incident.description)
        result['request_id'] = request_id
        log.info('investigation id=%s duration_ms=%d method=%s', request_id, (time.monotonic()-started)*1000, result.get('retrieval_method'))
        return result
    except Exception as exc:
        log.error('investigation failed id=%s error_type=%s', request_id, type(exc).__name__)
        # Never leak Azure/Databricks credentials or raw connector exceptions to callers.
        raise HTTPException(status_code=502, detail="Incident analysis failed; inspect server logs") from exc

@app.get("/")
def index():
    return FileResponse(Path(__file__).with_name("index.html"))
