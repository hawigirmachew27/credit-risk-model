"""
src/train.py
============
Model training, hyperparameter tuning, and MLflow experiment tracking.
Task 5: Model Training and Tracking

Author: Bati Bank Analytics Team
"""

import logging
import os
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE    = 0.2
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT   = "bati_bank_credit_risk"

# Default paths
DATA_DIR      = os.path.join("data", "processed")
DATA_PATH     = os.path.join(DATA_DIR, "model_dataset.csv")
MODEL_DIR     = os.path.join("data", "processed")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")

# Features used for training (scaled versions from data_processing pipeline)
FEATURE_COLS = [
    "Recency_scaled", "Frequency_scaled", "Monetary_scaled",
    "total_amount_scaled", "avg_amount_scaled", "std_amount_scaled",
    "txn_count_scaled", "debit_count_scaled", "debit_total_scaled",
    "fraud_rate", "debit_ratio", "unique_products_scaled",
    "unique_channels_scaled",
]
TARGET_COL = "is_high_risk"


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def load_processed_data(path: str) -> pd.DataFrame:
    """Load the processed customer-level dataset produced by data_processing.py."""
    logger.info(f"Loading processed dataset from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows")
    return df


def prepare_features(df: pd.DataFrame):
    """
    Select and validate feature columns.
    Falls back to unscaled columns if scaled versions are absent
    (supports running directly on the raw RFM + aggregate merge).
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    if len(available) < 3:
        # Fall back: use unscaled numeric columns and scale on the fly
        fallback = [
            "Recency", "Frequency", "Monetary",
            "total_amount", "avg_amount", "std_amount",
            "txn_count", "fraud_rate", "debit_ratio",
        ]
        available = [c for c in fallback if c in df.columns]
        logger.warning(f"Using fallback (unscaled) features: {available}")

    X = df[available].fillna(0)
    y = df[TARGET_COL]
    logger.info(f"Features selected: {list(X.columns)}")
    logger.info(f"Target distribution:\n{y.value_counts(normalize=True).to_string()}")
    return X, y


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute standard credit scoring evaluation metrics."""
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
    }


def save_model_joblib(model, path: str, model_name: str = ""):
    """Persist a fitted sklearn Pipeline to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"{'[' + model_name + '] ' if model_name else ''}Model saved → {path}")


def load_model_joblib(path: str):
    """Load a joblib-serialised model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model file found at: {path}")
    model = joblib.load(path)
    logger.info(f"Model loaded ← {path}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Model Definitions
# ──────────────────────────────────────────────────────────────────────────────

def get_models():
    """
    Return candidate models with hyperparameter grids for tuning.
    Each entry: (name, estimator, param_grid)
    """
    models = [
        (
            "LogisticRegression",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
                ("clf",     LogisticRegression(
                    random_state=RANDOM_STATE, max_iter=1000,
                    class_weight="balanced"
                )),
            ]),
            {
                "clf__C":       [0.01, 0.1, 1.0, 10.0],
                "clf__solver":  ["lbfgs", "liblinear"],
                "clf__penalty": ["l2"],
            },
        ),
        (
            "RandomForest",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf",     RandomForestClassifier(
                    random_state=RANDOM_STATE, class_weight="balanced"
                )),
            ]),
            {
                "clf__n_estimators":      [100, 200],
                "clf__max_depth":         [5, 10, None],
                "clf__min_samples_split": [2, 5],
            },
        ),
        (
            "GradientBoosting",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf",     GradientBoostingClassifier(
                    random_state=RANDOM_STATE
                )),
            ]),
            {
                "clf__n_estimators":  [100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth":     [3, 5],
                "clf__subsample":     [0.8, 1.0],
            },
        ),
    ]
    return models


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train_and_track(data_path: str = DATA_PATH):
    """
    Full training workflow:
    1. Load processed data from  data/processed/model_dataset.csv
    2. Split into train/test
    3. For each model: grid search + MLflow logging + per-model joblib save
    4. Register best model in MLflow Model Registry
    5. Save best model to  data/processed/trained/best_model.joblib
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = load_processed_data(data_path)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    best_run_id      = None
    best_roc_auc     = -1.0
    best_model_name  = None
    best_estimator_global = None

    for model_name, estimator, param_grid in get_models():
        logger.info(f"\n{'='*60}\nTraining: {model_name}\n{'='*60}")

        with mlflow.start_run(run_name=model_name) as run:
            # ── Hyperparameter search ──────────────────────────────────────
            grid_search = GridSearchCV(
                estimator,
                param_grid,
                cv=5,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=0,
                refit=True,
            )
            grid_search.fit(X_train, y_train)
            best_estimator = grid_search.best_estimator_

            # ── Evaluate on test set ───────────────────────────────────────
            y_pred = best_estimator.predict(X_test)
            y_prob = best_estimator.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test, y_pred, y_prob)

            # ── Log to MLflow ──────────────────────────────────────────────
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.log_param("model_type",   model_name)
            mlflow.log_param("train_size",   len(X_train))
            mlflow.log_param("test_size",    len(X_test))
            mlflow.log_param("cv_folds",     5)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.sklearn.log_model(best_estimator, artifact_path="model")

            logger.info(f"Best params : {grid_search.best_params_}")
            logger.info(f"Metrics     : {metrics}")

            # ── Save every model as its own joblib ─────────────────────────
            per_model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
            save_model_joblib(best_estimator, per_model_path, model_name)

            # ── Log joblib artifact path to MLflow ─────────────────────────
            mlflow.log_param("joblib_path", per_model_path)

            # ── Track best overall model ───────────────────────────────────
            if metrics["roc_auc"] > best_roc_auc:
                best_roc_auc          = metrics["roc_auc"]
                best_run_id           = run.info.run_id
                best_model_name       = model_name
                best_estimator_global = best_estimator

    # ── Register best model in MLflow Model Registry ──────────────────────
    logger.info(f"\nBest model: {best_model_name} (ROC-AUC={best_roc_auc:.4f})")
    model_uri  = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(model_uri, "bati_bank_credit_risk_champion")
    logger.info(f"Registered model version: {registered.version}")

    # ── Save best model as canonical joblib ───────────────────────────────
    save_model_joblib(best_estimator_global, BEST_MODEL_PATH, f"BEST ({best_model_name})")
    logger.info(
        f"\nJoblib files written to: {MODEL_DIR}/\n"
        f"  • best_model.joblib          ← champion ({best_model_name})\n"
        f"  • LogisticRegression.joblib\n"
        f"  • RandomForest.joblib\n"
        f"  • GradientBoosting.joblib"
    )

    return {
        "best_run_id":    best_run_id,
        "best_model_name": best_model_name,
        "best_roc_auc":   best_roc_auc,
        "best_model_path": BEST_MODEL_PATH,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    result = train_and_track(data_path)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Best model   : {result['best_model_name']}")
    print(f"  ROC-AUC      : {result['best_roc_auc']:.4f}")
    print(f"  MLflow run   : {result['best_run_id']}")
    print(f"  Saved to     : {result['best_model_path']}")
    print("="*60)