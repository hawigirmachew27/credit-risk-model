"""
src/data_processing.py
=====================
Full feature engineering pipeline for the Bati Bank credit risk model.
Task 3: Feature Engineering
Task 4: Proxy Target Variable Engineering (RFM-based K-Means clustering)

Author: Bati Bank Analytics Team
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


# ──────────────────────────────────────────────────────────────────────────────
# 1. Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw Xente transaction CSV and parse timestamps."""
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path, parse_dates=["TransactionStartTime"])
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. Feature Extraction — Time Features
# ──────────────────────────────────────────────────────────────────────────────

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from TransactionStartTime.
    Returns a copy with 5 new columns.
    """
    df = df.copy()
    df["txn_hour"]      = df["TransactionStartTime"].dt.hour
    df["txn_day"]       = df["TransactionStartTime"].dt.day
    df["txn_month"]     = df["TransactionStartTime"].dt.month
    df["txn_year"]      = df["TransactionStartTime"].dt.year
    df["txn_dayofweek"] = df["TransactionStartTime"].dt.dayofweek
    logger.info("Time features extracted: txn_hour, txn_day, txn_month, txn_year, txn_dayofweek")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. Feature Engineering — Aggregate Features
# ──────────────────────────────────────────────────────────────────────────────

def build_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build customer-level aggregate features from raw transaction history.
    Returns a customer-level DataFrame (one row per CustomerId).
    """
    logger.info("Building aggregate features per customer...")

    # Debit-only subset (positive amounts = customer purchases)
    debits = df[df["Amount"] > 0]

    agg = df.groupby("CustomerId").agg(
        total_amount      =("Amount",        "sum"),
        avg_amount        =("Amount",        "mean"),
        std_amount        =("Amount",        "std"),
        max_amount        =("Amount",        "max"),
        min_amount        =("Amount",        "min"),
        txn_count         =("TransactionId", "count"),
        unique_products   =("ProductId",     "nunique"),
        unique_channels   =("ChannelId",     "nunique"),
        unique_categories =("ProductCategory", "nunique"),
        fraud_count       =("FraudResult",   "sum"),
    ).reset_index()

    # Debit-specific aggregates (handle customers with no debits)
    debit_agg = debits.groupby("CustomerId").agg(
        debit_count  =("Amount", "count"),
        debit_total  =("Amount", "sum"),
        debit_avg    =("Amount", "mean"),
    ).reset_index()

    agg = agg.merge(debit_agg, on="CustomerId", how="left")

    # Ratio features
    agg["fraud_rate"]      = agg["fraud_count"] / agg["txn_count"]
    agg["debit_ratio"]     = agg["debit_count"]  / agg["txn_count"]
    agg["std_amount"]      = agg["std_amount"].fillna(0)
    agg["debit_count"]     = agg["debit_count"].fillna(0)
    agg["debit_total"]     = agg["debit_total"].fillna(0)
    agg["debit_avg"]       = agg["debit_avg"].fillna(0)
    agg["debit_ratio"]     = agg["debit_ratio"].fillna(0)

    logger.info(f"Aggregate features built: {agg.shape[1]} columns for {len(agg):,} customers")
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# 4. Feature Engineering — RFM
# ──────────────────────────────────────────────────────────────────────────────

def build_rfm_features(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """
    Compute Recency, Frequency, Monetary (RFM) per CustomerId.
    - Recency: days since last transaction (lower = more recent = better)
    - Frequency: total transaction count
    - Monetary: sum of positive (debit) amounts only

    Returns a customer-level DataFrame.
    """
    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    logger.info(f"Building RFM features. Snapshot date: {snapshot_date.date()}")

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency   =("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
            Frequency =("TransactionId",        "count"),
            Monetary  =("Amount",               lambda x: x[x > 0].sum()),
        )
        .reset_index()
    )
    logger.info(f"RFM table: {rfm.shape[0]:,} customers")
    return rfm


# ──────────────────────────────────────────────────────────────────────────────
# 5. Categorical Encoding
# ──────────────────────────────────────────────────────────────────────────────

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns using Label Encoding.
    One-Hot Encoding is skipped to avoid high dimensionality at the
    customer-level aggregation stage; WoE is handled separately if needed.
    """
    df = df.copy()
    cat_cols = ["ProductCategory", "ChannelId", "ProviderId"]

    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
            logger.info(f"Label-encoded '{col}' → '{col}_encoded'")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 6. Winsorization (Outlier Handling)
# ──────────────────────────────────────────────────────────────────────────────

def winsorize_features(df: pd.DataFrame, cols: list, lower: float = 0.01,
                       upper: float = 0.99) -> pd.DataFrame:
    """
    Cap numerical features at specified quantiles (Winsorization).
    Documented for Basel II compliance — boundaries are logged explicitly.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        df[col] = df[col].clip(lower=lo, upper=hi)
        logger.info(f"Winsorized '{col}': [{lo:.2f}, {hi:.2f}]")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 7. Sklearn Preprocessing Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def get_preprocessing_pipeline() -> Pipeline:
    """
    Return a fitted-ready sklearn Pipeline for numerical feature scaling.
    Steps: median imputation → StandardScaler.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    return pipeline


# ──────────────────────────────────────────────────────────────────────────────
# 8. Task 4 — Proxy Target Variable via K-Means RFM Clustering
# ──────────────────────────────────────────────────────────────────────────────

def build_proxy_target(rfm: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Task 4: Cluster customers using K-Means on scaled RFM features and
    assign a binary is_high_risk label.

    High-risk cluster = lowest Frequency AND lowest Monetary AND highest Recency
    (i.e., least engaged customers).

    Returns rfm DataFrame with added columns:
        - rfm_cluster   : cluster label (0, 1, 2)
        - is_high_risk  : 1 = high risk, 0 = low risk
    """
    logger.info("Building proxy target variable via K-Means RFM clustering...")

    rfm = rfm.copy()
    features = ["Recency", "Frequency", "Monetary"]

    # Scale RFM before clustering (critical — Monetary dominates unscaled)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[features])

    # K-Means clustering with fixed random seed for reproducibility
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    rfm["rfm_cluster"] = kmeans.fit_predict(rfm_scaled)

    # Identify high-risk cluster: highest mean Recency, lowest Frequency & Monetary
    cluster_summary = rfm.groupby("rfm_cluster")[features].mean()
    logger.info(f"Cluster summary:\n{cluster_summary.to_string()}")

    # Score each cluster: high recency = bad, high freq/monetary = good
    # Normalise each dimension to [0,1] then combine
    cs = cluster_summary.copy()
    cs["score"] = (
        (cs["Recency"]   - cs["Recency"].min())   / (cs["Recency"].max()   - cs["Recency"].min() + 1e-9)
        - (cs["Frequency"] - cs["Frequency"].min()) / (cs["Frequency"].max() - cs["Frequency"].min() + 1e-9)
        - (cs["Monetary"]  - cs["Monetary"].min())  / (cs["Monetary"].max()  - cs["Monetary"].min() + 1e-9)
    )
    high_risk_cluster = int(cs["score"].idxmax())
    logger.info(f"High-risk cluster identified: cluster {high_risk_cluster}")

    rfm["is_high_risk"] = (rfm["rfm_cluster"] == high_risk_cluster).astype(int)
    high_risk_pct = rfm["is_high_risk"].mean() * 100
    logger.info(
        f"Proxy target assigned: {rfm['is_high_risk'].sum():,} high-risk customers "
        f"({high_risk_pct:.1f}%)"
    )
    return rfm


# ──────────────────────────────────────────────────────────────────────────────
# 9. Master Pipeline — Raw CSV → Model-Ready Dataset
# ──────────────────────────────────────────────────────────────────────────────

def build_model_dataset(raw_path: str, output_path: str = None) -> pd.DataFrame:
    """
    End-to-end pipeline: raw CSV → processed customer-level dataset
    with RFM features, aggregate features, and is_high_risk target.

    Steps:
        1. Load raw data
        2. Extract time features
        3. Build aggregate features (customer level)
        4. Build RFM features
        5. Winsorize monetary features
        6. Assign proxy target via K-Means
        7. Merge all features into final dataset
        8. Apply imputation + scaling pipeline
        9. Save processed output (optional)

    Returns:
        model_df (pd.DataFrame): customer-level, model-ready dataset
    """
    logger.info("=== Starting end-to-end feature engineering pipeline ===")

    # Step 1 — Load
    df = load_raw_data(raw_path)

    # Step 2 — Time features (transaction level)
    df = extract_time_features(df)

    # Step 3 — Aggregate features (customer level)
    agg = build_aggregate_features(df)

    # Step 4 — RFM
    rfm = build_rfm_features(df)

    # Step 5 — Winsorize RFM monetary feature
    rfm = winsorize_features(rfm, cols=["Monetary", "Frequency"], lower=0.01, upper=0.99)

    # Step 6 — Proxy target
    rfm = build_proxy_target(rfm)

    # Step 7 — Merge
    model_df = rfm.merge(agg, on="CustomerId", how="left")

    # Step 8 — Numerical feature scaling
    numeric_cols = [
        "Recency", "Frequency", "Monetary",
        "total_amount", "avg_amount", "std_amount", "max_amount", "min_amount",
        "txn_count", "debit_count", "debit_total", "debit_avg",
        "fraud_rate", "debit_ratio", "unique_products", "unique_channels",
    ]
    numeric_cols = [c for c in numeric_cols if c in model_df.columns]

    pipeline = get_preprocessing_pipeline()
    scaled_array = pipeline.fit_transform(model_df[numeric_cols])
    scaled_df = pd.DataFrame(scaled_array, columns=[c + "_scaled" for c in numeric_cols],
                             index=model_df.index)
    model_df = pd.concat([model_df, scaled_df], axis=1)

    logger.info(f"Final dataset: {model_df.shape[0]:,} customers × {model_df.shape[1]} columns")
    logger.info(f"Target distribution:\n{model_df['is_high_risk'].value_counts(normalize=True).to_string()}")

    # Step 9 — Save
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        model_df.to_csv(output_path, index=False)
        logger.info(f"Processed dataset saved → {output_path}")

    logger.info("=== Pipeline complete ===")
    return model_df


if __name__ == "__main__":
    import sys
    raw  = sys.argv[1] if len(sys.argv) > 1 else "data/raw/data.csv"
    out  = sys.argv[2] if len(sys.argv) > 2 else "data/processed/model_dataset.csv"
    build_model_dataset(raw, out)
