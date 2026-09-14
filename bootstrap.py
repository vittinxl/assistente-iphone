from pathlib import Path

from config_loader import get_chat_backend
from duckdb_store import ensure_duckdb_database

ML_DATA = Path(__file__).resolve().parent / "data" / "ml"
ML_MODELS = ML_DATA / "models"

try:
    from txtai_store import ensure_txtai_index, get_txtai_path
except ImportError:
    def get_txtai_path():
        return None

    def ensure_txtai_index():
        return None


def main() -> None:
    ML_DATA.mkdir(parents=True, exist_ok=True)
    ML_MODELS.mkdir(parents=True, exist_ok=True)
    print("ml: pastas data/ml e data/ml/models prontas.")

    try:
        ensure_duckdb_database()
        print("duckdb: banco demo pronto.")
    except Exception as exc:
        print(f"AVISO duckdb: {exc}")

    try:
        get_txtai_path()
    except Exception:
        pass

    if get_chat_backend() == "flowise":
        print("chat: backend Flowise (indexacao txtai opcional na aba txtai).")
        return

    try:
        result = ensure_txtai_index()
        if result:
            print(
                f"txtai: indexados {result['chunks_indexed']} trechos "
                f"de {result['files_indexed']} arquivos."
            )
        else:
            print("txtai: indice ja existente ou indexacao adiada.")
    except Exception as exc:
        print(f"AVISO txtai (app sobe mesmo assim): {exc}")


if __name__ == "__main__":
    main()
