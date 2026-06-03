"""
src/api/main.py
===============
FastAPI REST API for serving Bati Bank credit risk predictions.
Task 6: Model Deployment

Endpoints:
    GET  /health          — Service health check
    POST /predict         — Score a single customer
    POST /predict/batch   — Score multiple customers

Author: Bati Bank Analytics Team
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.pydantic_models import (
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Path to the joblib model saved by train.py
# Override via env var:  MODEL_PATH=data/processed/best_model.joblib
MODEL_PATH    = os.getenv("MODEL_PATH", "data/processed/best_model.joblib")
MODEL_VERSION = "joblib"

# Global model holder
_model = None


def get_risk_band(score: int) -> str:
    """Map credit score to a human-readable risk band."""
    if score >= 750: return "VERY_LOW"
    if score >= 670: return "LOW"
    if score >= 580: return "MEDIUM"
    if score >= 500: return "HIGH"
    return "VERY_HIGH"


def load_model():
    """Load champion model from joblib file saved by train.py."""
    global _model
    if not os.path.exists(MODEL_PATH):
        logger.warning(
            f"Model file not found at '{MODEL_PATH}'. "
            "Run the training pipeline first (train.py), then restart the API."
        )
        _model = None
        return

    try:
        _model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from: {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model from '{MODEL_PATH}': {e}")
        _model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_model()
    yield
    logger.info("Shutting down credit risk API")


app = FastAPI(
    title="Bati Bank Credit Risk API",
    description=(
        "Scores customer credit risk for the Bati Bank × Xente BNPL product. "
        "Returns a risk probability, binary risk label, and credit score (300–850)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Service health check — returns model status."""
    return HealthResponse(
        status="ok" if _model is not None else "model_not_loaded",
        model=MODEL_PATH,
        version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
def predict(features: CustomerFeatures):
    """
    Score a single customer and return their credit risk profile.

    - **risk_probability**: 0 (safe) → 1 (likely default)
    - **risk_label**: 1 = high risk, 0 = low risk
    - **credit_score**: 300 (worst) → 850 (best)
    - **risk_band**: VERY_HIGH / HIGH / MEDIUM / LOW / VERY_LOW
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Ensure '{MODEL_PATH}' exists (run train.py first).",
        )

    try:
        df    = pd.DataFrame([features.model_dump()])
        proba = float(_model.predict_proba(df)[0, 1])
        label = int(proba >= 0.5)
        score = int(round(850 - proba * 550))
        band  = get_risk_band(score)

        logger.info(f"Prediction: prob={proba:.4f} label={label} score={score} band={band}")

        return PredictionResponse(
            risk_probability=round(proba, 4),
            risk_label=label,
            credit_score=score,
            risk_band=band,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Scoring"])
def predict_batch(customers: List[CustomerFeatures]):
    """
    Score multiple customers in a single request.
    Maximum 500 customers per request.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Ensure '{MODEL_PATH}' exists (run train.py first).",
        )
    if len(customers) > 500:
        raise HTTPException(status_code=400, detail="Maximum 500 customers per batch request.")

    try:
        df     = pd.DataFrame([c.model_dump() for c in customers])
        probas = _model.predict_proba(df)[:, 1]
        results = []
        for proba in probas:
            label = int(proba >= 0.5)
            score = int(round(850 - proba * 550))
            results.append(PredictionResponse(
                risk_probability=round(float(proba), 4),
                risk_label=label,
                credit_score=score,
                risk_band=get_risk_band(score),
            ))
        return results
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))