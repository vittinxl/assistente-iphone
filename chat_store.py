import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from text_utils import repair_mojibake


CHAT_STORE_PATH = Path(os.getenv("CHAT_STORE_PATH", Path(__file__).parent / ".chat_history.json"))
DEFAULT_CHAT_TITLE = "Novo chat"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_chat(title: str = DEFAULT_CHAT_TITLE) -> dict:
    now = _now_iso()
    return {
        "id": str(uuid4()),
        "title": title,
        "messages": [],
        "flowise_chat_id": None,
        "created_at": now,
        "updated_at": now,
    }


def _normalize_message(message: object) -> dict[str, str] | None:
    if not isinstance(message, dict):
        return None

    role = str(message.get("role", "")).strip()
    content = str(message.get("content", ""))
    if not role or not content.strip():
        return None

    return {"role": role, "content": repair_mojibake(content)}


def _normalize_chat(chat: object) -> dict:
    if not isinstance(chat, dict):
        return create_chat()

    created_at = str(chat.get("created_at") or _now_iso())
    updated_at = str(chat.get("updated_at") or created_at)
    flowise_chat_id = chat.get("flowise_chat_id")

    messages = []
    for message in chat.get("messages", []):
        normalized = _normalize_message(message)
        if normalized:
            messages.append(normalized)

    return {
        "id": str(chat.get("id") or uuid4()),
        "title": str(chat.get("title") or DEFAULT_CHAT_TITLE),
        "messages": messages,
        "flowise_chat_id": str(flowise_chat_id) if flowise_chat_id else None,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def load_chats() -> list[dict]:
    if not CHAT_STORE_PATH.is_file():
        return []

    try:
        with CHAT_STORE_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return []

    chats = data.get("chats", [])
    if not isinstance(chats, list):
        return []

    normalized_chats = [_normalize_chat(chat) for chat in chats]
    return sorted(normalized_chats, key=lambda chat: chat["updated_at"], reverse=True)


def save_chats(chats: list[dict]) -> None:
    CHAT_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalized_chats = [_normalize_chat(chat) for chat in chats]
    payload = {"chats": normalized_chats}
    temp_path = CHAT_STORE_PATH.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    # Replace the file atomically to avoid partial writes.
    temp_path.replace(CHAT_STORE_PATH)
