"""Chat direto com LM Studio (API OpenAI-compatível) — sem depender do Agent do Flowise."""

from __future__ import annotations

import requests

from config_loader import get_lmstudio_settings
from text_utils import repair_mojibake


class LmStudioError(Exception):
    pass


def check_server() -> tuple[bool, str]:
    settings = get_lmstudio_settings()
    try:
        response = requests.get(
            f"{settings.base_url.rstrip('/')}/models",
            timeout=5,
        )
        if response.ok:
            return True, "LM Studio respondendo"
        return False, f"HTTP {response.status_code}"
    except requests.RequestException as exc:
        return False, str(exc)


def list_models() -> list[str]:
    settings = get_lmstudio_settings()
    response = requests.get(f"{settings.base_url.rstrip('/')}/models", timeout=10)
    if not response.ok:
        raise LmStudioError(f"Nao foi possivel listar modelos: HTTP {response.status_code}")
    data = response.json()
    models = []
    for item in data.get("data", []):
        mid = item.get("id")
        if mid:
            models.append(mid)
    return models


def chat(
    user_message: str,
    history: list[dict[str, str]] | None = None,
    system_prompt: str | None = None,
) -> str:
    settings = get_lmstudio_settings()
    messages: list[dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for msg in history or []:
        role = "assistant" if msg.get("role") == "assistant" else "user"
        content = str(msg.get("content", ""))
        if content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": settings.model,
        "messages": messages,
        "temperature": settings.temperature,
        "stream": False,
    }

    url = f"{settings.base_url.rstrip('/')}/chat/completions"
    try:
        response = requests.post(url, json=payload, timeout=180)
    except requests.RequestException as exc:
        raise LmStudioError(
            f"Sem conexao com LM Studio em {settings.base_url}. "
            f"Abra o LM Studio, carregue um modelo e clique Start Server. ({exc})"
        ) from exc

    if not response.ok:
        text = (response.text or "")[:400]
        raise LmStudioError(f"LM Studio HTTP {response.status_code}: {text}")

    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LmStudioError(f"Resposta inesperada do LM Studio: {data}") from exc

    return repair_mojibake(content or "")
