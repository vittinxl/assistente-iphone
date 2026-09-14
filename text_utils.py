"""Correção de mojibake (UTF-8 lido como Latin-1)."""


def looks_like_mojibake(text: str) -> bool:
    if not text:
        return False
    markers = ("Ã", "Â", "â€", "ï¿½")
    return any(marker in text for marker in markers)


def repair_mojibake(text: str) -> str:
    """
    Converte sequências como 'informaÃ§Ãµes' -> 'informações'.
    Só altera o texto quando há indícios claros de mojibake.
    """
    if not text or not looks_like_mojibake(text):
        return text

    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text

    if looks_like_mojibake(repaired) and repaired != text:
        return text

    return repaired
