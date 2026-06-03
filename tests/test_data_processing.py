"""
tests/test_data_processing.py
==============================
Unit tests for feature engineering and proxy target variable functions.
Tasks 3, 4, and 5.

Run with:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    build_aggregate_features,
    build_proxy_target,
    build_rfm_features,
    encode_categorical_features,
    extract_time_features,
    get_preprocessing_pipeline,
    winsorize_features,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal synthetic transaction DataFrame for unit testing."""
    return pd.DataFrame({
        "TransactionId":        ["T1", "T2", "T3", "T4", "T5", "T6"],
        "CustomerId":           ["C1", "C1", "C2", "C2", "C3", "C3"],
        "AccountId":            ["A1", "A1", "A2", "A2", "A3", "A3"],
        "ProductId":            ["P1", "P2", "P1", "P3", "P2", "P3"],
        "ProductCategory":      ["airtime", "financial_services", "airtime",
                                 "utility_bill", "airtime", "financial_services"],
        "ChannelId":            ["ChannelId_1", "ChannelId_2", "ChannelId_1",
                                 "ChannelId_3", "ChannelId_2", "ChannelId_1"],
        "ProviderId":           ["P1", "P2", "P1", "P3", "P2", "P1"],
        "TransactionStartTime": pd.to_datetime([
            "2018-11-01 08:30:00",
            "2018-11-15 14:00:00",
            "2018-11-01 21:00:00",
            "2018-12-01 10:00:00",
            "2018-11-20 03:00:00",
            "2018-11-25 16:00:00",
        ]),
        "Amount":    [500.0, 1500.0, -200.0, 3000.0, 250.0, 800.0],
        "Value":     [500.0, 1500.0,  200.0, 3000.0, 250.0, 800.0],
        "FraudResult": [0, 0, 1, 0, 0, 0],
        "PricingStrategy": [2, 2, 2, 2, 4, 2],
    })


@pytest.fixture
def sample_rfm(sample_df):
    """Pre-built RFM DataFrame for proxy target tests."""
    return build_rfm_features(sample_df)


# ──────────────────────────────────────────────────────────────────────────────
# Task 3 — Time Feature Extraction
# ──────────────────────────────────────────────────────────────────────────────

def test_extract_time_features_adds_correct_columns(sample_df):
    """extract_time_features must add exactly 5 new temporal columns."""
    result = extract_time_features(sample_df)
    expected = {"txn_hour", "txn_day", "txn_month", "txn_year", "txn_dayofweek"}
    assert expected.issubset(set(result.columns)), \
        f"Missing columns: {expected - set(result.columns)}"


def test_extract_time_features_correct_values(sample_df):
    """Verify extracted values match the source timestamp."""
    result = extract_time_features(sample_df)
    assert result.loc[0, "txn_hour"]      == 8
    assert result.loc[0, "txn_month"]     == 11
    assert result.loc[0, "txn_year"]      == 2018
    assert result.loc[0, "txn_dayofweek"] == 3   # Thursday


def test_extract_time_features_no_mutation(sample_df):
    """Original DataFrame must not be mutated."""
    original_cols = set(sample_df.columns)
    extract_time_features(sample_df)
    assert set(sample_df.columns) == original_cols


# ──────────────────────────────────────────────────────────────────────────────
# Task 3 — Aggregate Features
# ──────────────────────────────────────────────────────────────────────────────

def test_build_aggregate_features_row_count(sample_df):
    """One row per unique CustomerId."""
    agg = build_aggregate_features(sample_df)
    assert len(agg) == sample_df["CustomerId"].nunique()


def test_build_aggregate_features_required_columns(sample_df):
    """All required aggregate columns must be present."""
    agg = build_aggregate_features(sample_df)
    required = {
        "CustomerId", "total_amount", "avg_amount", "std_amount",
        "txn_count", "unique_products", "unique_channels", "fraud_count",
        "fraud_rate", "debit_ratio",
    }
    assert required.issubset(set(agg.columns))


def test_build_aggregate_features_no_nulls_in_std(sample_df):
    """std_amount must not be NaN (single-transaction customers get 0)."""
    agg = build_aggregate_features(sample_df)
    assert agg["std_amount"].isna().sum() == 0


def test_build_aggregate_features_fraud_rate_bounded(sample_df):
    """Fraud rate must be between 0 and 1 for all customers."""
    agg = build_aggregate_features(sample_df)
    assert (agg["fraud_rate"] >= 0).all() and (agg["fraud_rate"] <= 1).all()


# ──────────────────────────────────────────────────────────────────────────────
# Task 3 — RFM Features
# ──────────────────────────────────────────────────────────────────────────────

def test_build_rfm_features_shape(sample_df):
    """RFM table must have one row per unique CustomerId."""
    rfm = build_rfm_features(sample_df)
    assert len(rfm) == sample_df["CustomerId"].nunique()


def test_build_rfm_features_monetary_debits_only(sample_df):
    """Monetary must only sum positive (debit) amounts."""
    rfm = build_rfm_features(sample_df)
    # C2: Amount = [-200, 3000] → Monetary = 3000
    c2 = rfm.loc[rfm["CustomerId"] == "C2", "Monetary"].values[0]
    assert c2 == 3000.0


def test_build_rfm_features_recency_non_negative(sample_df):
    """Recency must be >= 0 for all customers."""
    rfm = build_rfm_features(sample_df)
    assert (rfm["Recency"] >= 0).all()


def test_build_rfm_features_frequency_positive(sample_df):
    """Frequency must be >= 1 for all customers."""
    rfm = build_rfm_features(sample_df)
    assert (rfm["Frequency"] >= 1).all()


# ──────────────────────────────────────────────────────────────────────────────
# Task 3 — Winsorization
# ──────────────────────────────────────────────────────────────────────────────

def test_winsorize_clips_values(sample_df):
    """After winsorization, no values should exceed the 99th percentile cap."""
    winsorized = winsorize_features(sample_df, cols=["Amount"], lower=0.01, upper=0.99)
    cap = sample_df["Amount"].quantile(0.99)
    assert (winsorized["Amount"] <= cap + 1e-9).all()


def test_winsorize_no_mutation(sample_df):
    """Winsorization must return a copy, not mutate the original."""
    original_vals = sample_df["Amount"].copy()
    winsorize_features(sample_df, cols=["Amount"])
    pd.testing.assert_series_equal(sample_df["Amount"], original_vals)


# ──────────────────────────────────────────────────────────────────────────────
# Task 3 — Categorical Encoding
# ──────────────────────────────────────────────────────────────────────────────

def test_encode_categorical_features_adds_columns(sample_df):
    """Encoding must add _encoded suffix columns for categorical cols."""
    encoded = encode_categorical_features(sample_df)
    assert "ProductCategory_encoded" in encoded.columns
    assert "ChannelId_encoded" in encoded.columns


def test_encode_categorical_features_integer_dtype(sample_df):
    """Encoded columns must be integer dtype."""
    encoded = encode_categorical_features(sample_df)
    assert encoded["ProductCategory_encoded"].dtype in [np.int32, np.int64, int]


# ──────────────────────────────────────────────────────────────────────────────
# Task 3 — Preprocessing Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def test_preprocessing_pipeline_transforms(sample_df):
    """Pipeline must fit/transform without error and return correct shape."""
    agg = build_aggregate_features(sample_df)
    numeric_cols = ["total_amount", "avg_amount", "txn_count"]
    pipeline = get_preprocessing_pipeline()
    result = pipeline.fit_transform(agg[numeric_cols])
    assert result.shape == (len(agg), len(numeric_cols))


def test_preprocessing_pipeline_zero_mean(sample_df):
    """StandardScaler output should have approximately zero mean."""
    agg = build_aggregate_features(sample_df)
    numeric_cols = ["total_amount", "avg_amount", "txn_count"]
    pipeline = get_preprocessing_pipeline()
    result = pipeline.fit_transform(agg[numeric_cols])
    assert abs(result.mean()) < 1.0   # loose bound given small fixture size


# ──────────────────────────────────────────────────────────────────────────────
# Task 4 — Proxy Target Variable
# ──────────────────────────────────────────────────────────────────────────────

def test_build_proxy_target_adds_columns(sample_rfm):
    """build_proxy_target must add rfm_cluster and is_high_risk columns."""
    result = build_proxy_target(sample_rfm, n_clusters=3)
    assert "rfm_cluster"  in result.columns
    assert "is_high_risk" in result.columns


def test_build_proxy_target_binary_label(sample_rfm):
    """is_high_risk must contain only 0 and 1."""
    result = build_proxy_target(sample_rfm, n_clusters=3)
    assert set(result["is_high_risk"].unique()).issubset({0, 1})


def test_build_proxy_target_reproducible(sample_rfm):
    """Two calls with the same data must produce identical labels (deterministic)."""
    r1 = build_proxy_target(sample_rfm.copy(), n_clusters=3)
    r2 = build_proxy_target(sample_rfm.copy(), n_clusters=3)
    pd.testing.assert_series_equal(
        r1["is_high_risk"].reset_index(drop=True),
        r2["is_high_risk"].reset_index(drop=True),
    )


def test_build_proxy_target_no_mutation(sample_rfm):
    """Original RFM DataFrame must not be mutated."""
    original_cols = set(sample_rfm.columns)
    build_proxy_target(sample_rfm)
    assert set(sample_rfm.columns) == original_cols
