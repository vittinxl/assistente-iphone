import requests

from auth import get_headers
from config_loader import get_flowise_settings
from text_utils import repair_mojibake


class FlowiseAPIError(Exception):
    pass


def _format_http_error(response: requests.Response) -> str:
    try:
        body = response.json()
        if isinstance(body, dict):
            for key in ("message", "error", "detail"):
                value = body.get(key)
                if value:
                    return f"{response.status_code}: {value}"
        return f"{response.status_code}: {body}"
    except ValueError:
        text = (response.text or "").strip()
        return f"{response.status_code}: {text[:500] if text else response.reason}"


def query(payload: dict) -> dict:
    settings = get_flowise_settings()
    response = requests.post(
        settings.api_url,
        headers=get_headers(),
        json=payload,
        timeout=180,
    )
    if not response.ok:
        raise FlowiseAPIError(_format_http_error(response))
    return response.json()


def extract_reply(data: dict) -> str:
    if isinstance(data.get("text"), str):
        return repair_mojibake(data["text"])
    if isinstance(data.get("answer"), str):
        return repair_mojibake(data["answer"])
    return repair_mojibake(str(data))
