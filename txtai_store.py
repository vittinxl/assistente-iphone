import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import google.generativeai as genai
from txtai.embeddings import Embeddings

from text_utils import repair_mojibake

APP_DIR = Path(__file__).parent
STORE_DOCS_DIR = APP_DIR / "store_docs"
TXTAI_PATH = Path(os.getenv("TXTAI_PATH", APP_DIR / "data" / "txtai"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004").strip()

# Configure the Gemini API if key is present
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

@dataclass(frozen=True)
class StoreDocument:
    file_name: str
    path: Path
    title: str

def get_txtai_path() -> Path:
    TXTAI_PATH.mkdir(parents=True, exist_ok=True)
    return TXTAI_PATH

def get_store_documents() -> list[StoreDocument]:
    documents: list[StoreDocument] = []
    for path in sorted(STORE_DOCS_DIR.glob("*.md")):
        title = path.stem.replace("_", " ").title()
        documents.append(StoreDocument(file_name=path.name, path=path, title=title))
    return documents

def has_gemini_api_key() -> bool:
    return bool(GEMINI_API_KEY)

def get_gemini_embedding_model() -> str:
    return GEMINI_EMBEDDING_MODEL

def get_gemini_embeddings(texts):
    if not has_gemini_api_key():
        raise ValueError("GEMINI_API_KEY nao configurada. Defina a chave por variavel de ambiente ou arquivo .env.")
    
    # Standardize input for batching and formatting
    is_single = isinstance(texts, str)
    contents_list = [texts] if is_single else list(texts)
    
    all_embeddings = []

    for text in contents_list:
        response = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document",
        )
        emb = response.get("embedding", [])
        if not emb:
            continue
        if isinstance(emb[0], (int, float)):
            all_embeddings.append(emb)
        else:
            all_embeddings.extend(emb)
                
    if is_single:
        return np.array(all_embeddings[0], dtype=np.float32)
    else:
        return np.array(all_embeddings, dtype=np.float32)

_embeddings_instance = None


def reset_embeddings_client() -> None:
    global _embeddings_instance
    _embeddings_instance = None


def _get_embeddings_client() -> Embeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = Embeddings({
            "transform": get_gemini_embeddings,
            "backend": "numpy",
            "content": True,
        })
        index_path = get_txtai_path()
        try:
            if index_path.is_dir() and any(index_path.iterdir()):
                _embeddings_instance.load(str(index_path))
        except (OSError, Exception):
            reset_embeddings_client()
            _embeddings_instance = Embeddings({
                "transform": get_gemini_embeddings,
                "backend": "numpy",
                "content": True,
            })
    return _embeddings_instance


def is_index_ready() -> bool:
    return get_txtai_status()["indexed"]

def _chunk_text(text: str) -> list[str]:
    CHUNK_SIZE = int(os.getenv("TXTAI_CHUNK_SIZE", os.getenv("CHROMA_CHUNK_SIZE", "800")))
    CHUNK_OVERLAP = int(os.getenv("TXTAI_CHUNK_OVERLAP", os.getenv("CHROMA_CHUNK_OVERLAP", "120")))
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    while start < text_length:
        chunk = normalized[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks

def get_txtai_status() -> dict:
    index_path = get_txtai_path()
    has_index = False
    count = 0
    try:
        if any(index_path.iterdir()):
            has_index = True
    except (OSError, Exception):
        pass
    
    if has_index:
        try:
            embeddings = _get_embeddings_client()
            count = embeddings.count()
        except Exception:
            pass
            
    return {
        "path": str(index_path),
        "collection_name": "txtai_numpy_index",
        "has_openai_api_key": has_gemini_api_key(),  # Keep key name for UI compatibility or update in app.py
        "embedding_model": GEMINI_EMBEDDING_MODEL,
        "collections": ["txtai_numpy_index"] if has_index else [],
        "document_files": len(get_store_documents()),
        "chunk_count": count,
        "indexed": count > 0,
    }

def rebuild_index() -> dict:
    if not has_gemini_api_key():
        raise ValueError("GEMINI_API_KEY nao configurada. Sem ela nao e possivel indexar documentos no txtai.")

    reset_embeddings_client()
    embeddings = _get_embeddings_client()
    
    documents_to_index = []
    for store_document in get_store_documents():
        content = store_document.path.read_text(encoding="utf-8")
        for index, chunk in enumerate(_chunk_text(content)):
            chunk_id = hashlib.sha1(f"{store_document.file_name}:{index}:{chunk}".encode("utf-8")).hexdigest()
            documents_to_index.append((
                chunk_id,
                {
                    "text": chunk,
                    "source_file": store_document.file_name,
                    "title": store_document.title,
                    "chunk_index": index
                }
            ))

    if documents_to_index:
        embeddings.index(documents_to_index)
        embeddings.save(str(get_txtai_path()))
    
    return {
        "files_indexed": len(get_store_documents()),
        "chunks_indexed": len(documents_to_index),
        "collection_name": "txtai_numpy_index",
    }

def query_index(query_text: str, n_results: int = 5) -> list[dict]:
    if not query_text.strip():
        raise ValueError("Digite uma consulta para pesquisar no txtai.")
    if not has_gemini_api_key():
        raise ValueError("GEMINI_API_KEY nao configurada. Sem ela nao e possivel consultar semanticamente.")
    if not is_index_ready():
        return []

    embeddings = _get_embeddings_client()
    if embeddings.count() == 0:
        return []

    safe_results = max(1, min(n_results, 10))
    results = embeddings.search(query_text, limit=safe_results)
    
    output = []
    for r in results:
        output.append({
            "text": repair_mojibake(r.get("text", "")),
            "title": r.get("title", "Documento sem titulo"),
            "source_file": r.get("source_file", "n/d"),
            "chunk_index": r.get("chunk_index", "n/d"),
            "score": r.get("score", 0.0)
        })
    return output

def peek_index(limit: int = 10) -> list[dict]:
    embeddings = _get_embeddings_client()
    safe_limit = max(1, min(limit, 50))
    try:
        results = embeddings.search(f"SELECT id, text, source_file, title, chunk_index FROM txtai LIMIT {safe_limit}")
        output = []
        for r in results:
            output.append({
                "id": r.get("id", ""),
                "text": r.get("text", ""),
                "title": r.get("title", ""),
                "source_file": r.get("source_file", ""),
                "chunk_index": r.get("chunk_index", "")
            })
        return output
    except Exception:
        return []


def ensure_txtai_index() -> dict | None:
    """Cria o indice na inicializacao se ainda nao existir."""
    if is_index_ready():
        return None
    if not has_gemini_api_key():
        return None
    return rebuild_index()
