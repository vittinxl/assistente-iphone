from config_loader import get_flowise_settings


def get_headers() -> dict[str, str]:
    token = get_flowise_settings().bearer_token
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}
