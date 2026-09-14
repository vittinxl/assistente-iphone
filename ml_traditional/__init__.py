"""ML tradicional — FLAML AutoML + UCI 350 (Default of Credit Card Clients)."""

from ml_traditional.data_loader import load_raw_dataframe, prepare_features
from ml_traditional.dataset_catalog import COLUMN_DOCS, TARGET_COLUMN, UCI_PAGE_URL

__all__ = [
    "COLUMN_DOCS",
    "TARGET_COLUMN",
    "UCI_PAGE_URL",
    "load_raw_dataframe",
    "prepare_features",
    "render_ml_tab",
]


def __getattr__(name: str):
    """Import tardio de trainer/ui (evita carregar LightGBM ao abrir o app)."""
    if name in {
        "DEFAULT_MODEL_NAME",
        "FlamlConfig",
        "ModelBundle",
        "load_model_bundle",
        "predict_new_data",
        "save_model_bundle",
        "train_flaml_model",
    }:
        from ml_traditional import trainer as _trainer

        return getattr(_trainer, name)
    if name == "render_ml_tab":
        from ml_traditional.ui import render_ml_tab

        return render_ml_tab
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
