"""
src/api/pydantic_models.py
===========================
Request and response schemas for the credit risk prediction API.
Task 6: Model Deployment

Author: Bati Bank Analytics Team
"""

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """
    Input features for a single customer to be scored.
    All values are the scaled/engineered features produced by data_processing.py.
    """
    # RFM features (scaled)
    Recency_scaled:   float = Field(..., description="Days since last transaction (z-score scaled)")
    Frequency_scaled: float = Field(..., description="Transaction count (z-score scaled)")
    Monetary_scaled:  float = Field(..., description="Total debit spend (z-score scaled)")

    # Aggregate features (scaled)
    total_amount_scaled:  float = Field(0.0, description="Sum of all transaction amounts (scaled)")
    avg_amount_scaled:    float = Field(0.0, description="Mean transaction amount (scaled)")
    std_amount_scaled:    float = Field(0.0, description="Std dev of transaction amounts (scaled)")
    txn_count_scaled:     float = Field(0.0, description="Total transaction count (scaled)")
    debit_count_scaled:   float = Field(0.0, description="Debit transaction count (scaled)")
    debit_total_scaled:   float = Field(0.0, description="Sum of debit amounts (scaled)")
    unique_products_scaled: float = Field(0.0, description="Unique products purchased (scaled)")
    unique_channels_scaled: float = Field(0.0, description="Unique channels used (scaled)")

    # Ratio features (not scaled — already [0,1])
    fraud_rate:  float = Field(0.0, ge=0.0, le=1.0, description="Fraction of transactions flagged as fraud")
    debit_ratio: float = Field(0.0, ge=0.0, le=1.0, description="Fraction of transactions that are debits")

    class Config:
        json_schema_extra = {
            "example": {
                "Recency_scaled":   -0.5,
                "Frequency_scaled":  1.2,
                "Monetary_scaled":   0.8,
                "total_amount_scaled": 0.7,
                "avg_amount_scaled":   0.3,
                "std_amount_scaled":   0.4,
                "txn_count_scaled":    1.1,
                "debit_count_scaled":  0.9,
                "debit_total_scaled":  0.7,
                "unique_products_scaled": 0.2,
                "unique_channels_scaled": 0.1,
                "fraud_rate":  0.0,
                "debit_ratio": 0.85,
            }
        }


class PredictionResponse(BaseModel):
    """Risk prediction result for a single customer."""
    risk_probability: float = Field(..., description="Probability of default (0–1)")
    risk_label:       int   = Field(..., description="1 = high risk, 0 = low risk")
    credit_score:     int   = Field(..., description="Credit score 300–850 (higher = safer)")
    risk_band:        str   = Field(..., description="Risk band: VERY_HIGH / HIGH / MEDIUM / LOW / VERY_LOW")


class HealthResponse(BaseModel):
    status:  str
    model:   str
    version: str
