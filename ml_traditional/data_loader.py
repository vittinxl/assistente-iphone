"""Download e carga do dataset UCI 350."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from ml_traditional.dataset_catalog import (
    BETA_ANON_RENAME,
    COLUMN_DOCS,
    TARGET_COLUMN,
    TARGET_COLUMN_RAW,
    UCI_BETA_ZIP_URL,
    UCI_PAGE_URL,
    UCI_XLS_URL,
    UCI_ZIP_URL,
    column_dictionary_rows,
)

CSV_FALLBACK_URL = (
    "https://raw.githubusercontent.com/narayana36/Credit-Card-Defaulter-Prediction-Project/"
    "master/UCI_Credit_Card.csv"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ml"
CACHE_PARQUET = DATA_DIR / "credit_default_clients.parquet"
CACHE_XLS = DATA_DIR / "default_of_credit_card_clients.xls"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [str(c).strip() for c in renamed.columns]
    anon_map = {k: v for k, v in BETA_ANON_RENAME.items() if k in renamed.columns}
    if anon_map:
        renamed = renamed.rename(columns=anon_map)
    if TARGET_COLUMN_RAW in renamed.columns:
        renamed = renamed.rename(columns={TARGET_COLUMN_RAW: TARGET_COLUMN})
    if "Y" in renamed.columns and TARGET_COLUMN not in renamed.columns:
        renamed = renamed.rename(columns={"Y": TARGET_COLUMN})
    return renamed


def _download_bytes(url: str, timeout: int = 120) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def _load_from_xls_bytes(content: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(content), header=1)


def _load_zip_xls(zip_bytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        xls_names = [n for n in zf.namelist() if n.lower().endswith(".xls")]
        if not xls_names:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if csv_names:
                return _normalize_columns(pd.read_csv(io.BytesIO(zf.read(csv_names[0]))))
            raise FileNotFoundError("Nenhum .xls/.csv no ZIP do UCI.")
        xls_bytes = zf.read(xls_names[0])
    CACHE_XLS.write_bytes(xls_bytes)
    return _normalize_columns(_load_from_xls_bytes(xls_bytes))


def _load_from_uci_sources() -> pd.DataFrame:
    try:
        zip_bytes = _download_bytes(UCI_BETA_ZIP_URL)
        return _load_zip_xls(zip_bytes)
    except (requests.RequestException, ValueError, zipfile.BadZipFile, OSError, FileNotFoundError):
        pass

    try:
        xls_bytes = _download_bytes(UCI_XLS_URL)
        CACHE_XLS.write_bytes(xls_bytes)
        return _normalize_columns(_load_from_xls_bytes(xls_bytes))
    except (requests.RequestException, ValueError, zipfile.BadZipFile, OSError):
        pass

    try:
        zip_bytes = _download_bytes(UCI_ZIP_URL)
        return _load_zip_xls(zip_bytes)
    except (requests.RequestException, ValueError, zipfile.BadZipFile, OSError, FileNotFoundError):
        pass

    csv_bytes = _download_bytes(CSV_FALLBACK_URL)
    return _normalize_columns(pd.read_csv(io.BytesIO(csv_bytes)))


def download_dataset(force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_PARQUET.is_file() and not force:
        return CACHE_PARQUET

    if CACHE_XLS.is_file() and not force:
        df = _normalize_columns(_load_from_xls_bytes(CACHE_XLS.read_bytes()))
    else:
        df = _load_from_uci_sources()

    df.to_parquet(CACHE_PARQUET, index=False)
    return CACHE_PARQUET


def load_raw_dataframe(force_download: bool = False) -> pd.DataFrame:
    if force_download and CACHE_PARQUET.is_file():
        CACHE_PARQUET.unlink(missing_ok=True)
    path = download_dataset(force=force_download)
    return pd.read_parquet(path)


def prepare_features(
    df: pd.DataFrame,
    drop_columns: list[str],
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    work = df.copy()
    drop_set = {c for c in drop_columns if c in work.columns and c != target_column}
    feature_columns = [c for c in work.columns if c not in drop_set and c != target_column]

    X = work[feature_columns].copy()
    y = work[target_column].astype(int)

    for col in feature_columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")

    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    return X, y, feature_columns


def get_column_dictionary() -> pd.DataFrame:
    """Retorna o dicionário de colunas embutido no projeto."""
    return pd.DataFrame(column_dictionary_rows())


def build_inference_sample(df: pd.DataFrame, n_rows: int = 5) -> pd.DataFrame:
    """Linhas do dataset sem a coluna alvo — para simular predição em dados novos."""
    cols = [c for c in df.columns if c != TARGET_COLUMN]
    return df[cols].head(n_rows).copy()


def dataset_metadata() -> dict:
    return {
        "nome": "Default of Credit Card Clients",
        "uci_id": 350,
        "url": UCI_PAGE_URL,
        "linhas": 30_000,
        "alvo": TARGET_COLUMN,
        "colunas_documentadas": len(COLUMN_DOCS),
    }
