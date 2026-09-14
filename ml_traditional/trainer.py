"""Treino FLAML, persistência pickle e predição em novos dados."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ml_traditional.data_loader import prepare_features
from ml_traditional.dataset_catalog import TARGET_COLUMN

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "ml" / "models"
DEFAULT_MODEL_NAME = "credit_default_flaml.pkl"
DEFAULT_META_NAME = "credit_default_flaml.meta.json"


@dataclass
class FlamlConfig:
    time_budget: int = 120
    metric: str = "roc_auc"
    estimator_list: tuple[str, ...] = ("lgbm", "xgboost", "rf", "extra_tree")
    ensemble: bool = True
    n_splits: int = 5
    seed: int = 42
    eval_method: str = "cv"
    early_stop: bool = True
    log_training_metric: bool = True
    min_sample_split: int = 2
    min_child_samples: int = 20
    learning_rate: float = 0.1
    n_estimators: int = 200
    max_depth: int = 0  # 0 = sem limite explícito no FLAML para alguns estimadores


@dataclass
class ModelBundle:
    automl: Any
    feature_columns: list[str]
    dropped_columns: list[str]
    target_column: str
    flaml_config: dict[str, Any]
    best_estimator: str
    metrics: dict[str, Any]
    trained_at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def train_flaml_model(
    df: pd.DataFrame,
    drop_columns: list[str],
    config: FlamlConfig,
    test_size: float = 0.2,
) -> tuple[ModelBundle, pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    X, y, feature_columns = prepare_features(df, drop_columns=drop_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=config.seed,
        stratify=y,
    )

    automl = AutoML()
    fit_kwargs: dict[str, Any] = {
        "X_train": X_train,
        "y_train": y_train,
        "task": "classification",
        "metric": config.metric,
        "time_budget": config.time_budget,
        "eval_method": config.eval_method,
        "n_splits": config.n_splits,
        "seed": config.seed,
        "ensemble": config.ensemble,
        "estimator_list": list(config.estimator_list),
        "early_stop": config.early_stop,
        "log_training_metric": config.log_training_metric,
        "custom_hp": {
            "lgbm": {
                "n_estimators": {"domain": max(50, config.n_estimators // 2), "high": config.n_estimators},
                "learning_rate": {"domain": config.learning_rate / 4, "high": config.learning_rate},
                "min_child_samples": {"domain": config.min_child_samples, "high": config.min_child_samples * 2},
            },
            "xgboost": {
                "n_estimators": {"domain": max(50, config.n_estimators // 2), "high": config.n_estimators},
                "learning_rate": {"domain": config.learning_rate / 4, "high": config.learning_rate},
                "max_depth": {"domain": 3, "high": max(6, config.max_depth or 12)},
            },
            "rf": {
                "n_estimators": {"domain": max(50, config.n_estimators // 2), "high": config.n_estimators},
                "min_samples_split": {"domain": config.min_sample_split, "high": config.min_sample_split + 10},
            },
        },
    }
    automl.fit(**fit_kwargs)

    y_pred = automl.predict(X_test)
    y_proba = automl.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
    }

    bundle = ModelBundle(
        automl=automl,
        feature_columns=feature_columns,
        dropped_columns=[c for c in drop_columns if c in df.columns],
        target_column=TARGET_COLUMN,
        flaml_config=asdict(config),
        best_estimator=str(getattr(automl, "best_estimator", "unknown")),
        metrics=metrics,
        trained_at=_now_iso(),
    )
    return bundle, X_test, y_test, y_pred, y_proba


def save_model_bundle(bundle: ModelBundle, model_name: str = DEFAULT_MODEL_NAME) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / model_name
    meta_path = MODELS_DIR / DEFAULT_META_NAME

    joblib.dump(bundle, model_path)

    meta = {
        "model_file": model_name,
        "feature_columns": bundle.feature_columns,
        "dropped_columns": bundle.dropped_columns,
        "target_column": bundle.target_column,
        "best_estimator": bundle.best_estimator,
        "metrics": bundle.metrics,
        "flaml_config": bundle.flaml_config,
        "trained_at": bundle.trained_at,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return model_path


def load_model_bundle(model_name: str = DEFAULT_MODEL_NAME) -> ModelBundle:
    model_path = MODELS_DIR / model_name
    if not model_path.is_file():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")
    bundle = joblib.load(model_path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError("Arquivo pickle nao contem um ModelBundle valido.")
    return bundle


def load_meta(model_name: str = DEFAULT_MODEL_NAME) -> dict[str, Any]:
    meta_path = MODELS_DIR / DEFAULT_META_NAME
    if meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    bundle = load_model_bundle(model_name)
    return {
        "feature_columns": bundle.feature_columns,
        "dropped_columns": bundle.dropped_columns,
        "metrics": bundle.metrics,
        "best_estimator": bundle.best_estimator,
        "trained_at": bundle.trained_at,
    }


def validate_inference_columns(df: pd.DataFrame, bundle: ModelBundle) -> dict[str, list[str]]:
    expected = set(bundle.feature_columns)
    present = set(df.columns)
    return {
        "missing": sorted(expected - present),
        "extra": sorted(present - expected - {bundle.target_column}),
        "expected": bundle.feature_columns,
    }


def load_model_from_path(path: str | Path) -> ModelBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError("Arquivo pickle nao contem um ModelBundle valido.")
    return bundle


def align_inference_frame(df: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    aligned = df.copy()
    for col in bundle.feature_columns:
        if col not in aligned.columns:
            aligned[col] = 0
    aligned = aligned[bundle.feature_columns]
    for col in bundle.feature_columns:
        if aligned[col].dtype == "object":
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
    if aligned.isna().any().any():
        aligned = aligned.fillna(0)
    return aligned


def predict_new_data(
    df: pd.DataFrame,
    bundle: ModelBundle,
    proba_threshold: float | None = None,
) -> pd.DataFrame:
    check = validate_inference_columns(df, bundle)
    if len(check["missing"]) > len(bundle.feature_columns) // 2:
        raise ValueError(
            f"Muitas colunas faltando ({len(check['missing'])}). "
            f"Esperadas: {', '.join(bundle.feature_columns[:8])}..."
        )

    X = align_inference_frame(df, bundle)
    probas = bundle.automl.predict_proba(X)[:, 1]
    if proba_threshold is not None:
        labels = (probas >= proba_threshold).astype(int)
    else:
        labels = bundle.automl.predict(X)
    output = df.copy()
    output["pred_default"] = labels
    output["proba_default"] = probas
    output["colunas_faltantes_preenchidas"] = ", ".join(check["missing"]) or "(nenhuma)"
    return output
