from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.backend.schemas import AnalyzeRequest, AnalyzeResponse
from src.anna import ANNA


app = FastAPI(
    title="ANNA API",
    description="AI customer support intelligence for Uber Twitter support.",
    version="0.1.0",
) 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_anna() -> ANNA:
    return ANNA(top_k=5)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "ANNA API",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "anna-api",
    }


@app.get("/api/config")
def config() -> dict:
    anna = get_anna()

    return {
        "brand": "Uber",
        "agent": "ANNA",
        "top_k": anna.top_k,
        "capabilities": {
            "intent_classification": True,
            "historical_retrieval": True,
            "grounded_generation": True,
            "risk_detection": True,
            "automatic_escalation": True,
        },
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    try:
        result = get_anna().analyze(message)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"ANNA analysis failed: {exc}",
        ) from exc

    return AnalyzeResponse(**result.to_dict())
