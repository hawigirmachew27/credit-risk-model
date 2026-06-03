"""
src/predict.py
==============
Inference utilities — load champion model from MLflow registry and score customers.
Task 5 / Task 6: Inference

Author: Bati Bank Analytics Team
"""

import logging
import os

import mlflow.sklearn
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MODEL_NAME  = "bati_bank_credit_risk_champion"
MODEL_STAGE = "None"   # "Production" once promoted in the registry


def load_model(model_name: str = MODEL_NAME, stage: str = MODEL_STAGE):
    """Load the registered champion model from MLflow Model Registry."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    model_uri = f"models:/{model_name}/latest"
    logger.info(f"Loading model from MLflow registry: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    logger.info("Model loaded successfully")
    return model


def predict_risk(model, features: pd.DataFrame) -> pd.DataFrame:
    """
    Run inference on a feature DataFrame.

    Returns:
        DataFrame with columns:
            - risk_probability : P(is_high_risk=1)
            - risk_label       : 1 = high risk, 0 = low risk
            - credit_score     : scaled score 300–850 (higher = lower risk)
    """
    features = features.fillna(0)
    proba = model.predict_proba(features)[:, 1]
    labels = (proba >= 0.5).astype(int)

    # Credit score: linearly maps risk_probability [0,1] → score [850,300]
    # High risk probability → low credit score (industry convention)
    credit_scores = np.round(850 - (proba * 550)).astype(int)

    return pd.DataFrame({
        "risk_probability": np.round(proba, 4),
        "risk_label":       labels,
        "credit_score":     credit_scores,
    })


def score_single_customer(model, customer_features: dict) -> dict:
    """
    Score a single customer given a dict of features.
    Used by the FastAPI /predict endpoint.
    """
    df = pd.DataFrame([customer_features])
    result = predict_risk(model, df)
    return result.iloc[0].to_dict()
