import os
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore


CONFIG_PATH = Path(os.getenv("CONFIG_PATH", Path(__file__).parent / "config.toml"))


@dataclass(frozen=True)
class GeminiSettings:
    model: str
    embedding_model: str


@dataclass(frozen=True)
class FlowiseSettings:
    api_url: str
    bearer_token: str


@dataclass(frozen=True)
class LmStudioSettings:
    base_url: str
    model: str
    temperature: float


def _load_toml() -> dict:
    if not CONFIG_PATH.is_file():
        return {}
    with CONFIG_PATH.open("rb") as file:
        return tomllib.load(file)


def get_chat_backend() -> str:
    data = _load_toml()
    app_cfg = data.get("app", {})
    backend = str(app_cfg.get("chat_backend", "flowise")).strip().lower()
    env_backend = os.getenv("CHAT_BACKEND", "").strip().lower()
    return env_backend or backend or "flowise"


def get_flowise_settings() -> FlowiseSettings:
    data = _load_toml()
    flowise_cfg = data.get("flowise", {})

    api_url = os.getenv("FLOWISE_API_URL", "").strip() or str(flowise_cfg.get("api_url", "")).strip()
    bearer_token = os.getenv("FLOWISE_BEARER_TOKEN", "").strip() or str(
        flowise_cfg.get("bearer_token", "")
    ).strip()

    if not api_url:
        raise ValueError(
            "FLOWISE_API_URL nao configurada. Defina em config.toml na secao [flowise]."
        )

    return FlowiseSettings(api_url=api_url, bearer_token=bearer_token)


def get_lmstudio_settings() -> LmStudioSettings:
    data = _load_toml()
    lm_cfg = data.get("lmstudio", {})

    base_url = os.getenv("LMSTUDIO_BASE_URL", "").strip() or str(
        lm_cfg.get("base_url", "http://host.docker.internal:1234/v1")
    ).strip()
    model = os.getenv("LMSTUDIO_MODEL", "").strip() or str(
        lm_cfg.get("model", "local-model")
    ).strip()
    temp_raw = os.getenv("LMSTUDIO_TEMPERATURE", "").strip() or str(lm_cfg.get("temperature", 0.7))
    try:
        temperature = float(temp_raw)
    except ValueError:
        temperature = 0.7

    if not base_url:
        raise ValueError("LM Studio base_url nao configurada em config.toml [lmstudio].")

    return LmStudioSettings(base_url=base_url, model=model, temperature=temperature)


def get_settings() -> GeminiSettings:
    data = _load_toml()
    gemini_cfg = data.get("gemini", {})

    model = str(gemini_cfg.get("model", "gemini-2.0-flash")).strip()
    embedding_model = str(gemini_cfg.get("embedding_model", "models/text-embedding-004")).strip()

    env_model = os.getenv("GEMINI_MODEL", "").strip()
    env_embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "").strip()
    if env_model:
        model = env_model
    if env_embedding_model:
        embedding_model = env_embedding_model

    return GeminiSettings(model=model, embedding_model=embedding_model)
