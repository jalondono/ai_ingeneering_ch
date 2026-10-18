import os

import pytest

from backend import settings

VARIABLES = [
    "LLM_PROVIDER",
    "LLM_FALLBACK_PROVIDER",
    "LLM_TEMPERATURE",
    "LLM_MAX_TOKENS",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_MODEL",
    "ANTHROPIC_MODEL",
    "GEMINI_MODEL",
]


@pytest.fixture(autouse=True)
def entorno_limpio(monkeypatch):
    """Cada test arranca sin variables del proyecto y sin terminal interactiva.

    Se restaura os.environ completo al final porque cargar_entorno() y
    obtener_api_key() escriben en él directamente.
    """
    original = os.environ.copy()
    for variable in VARIABLES:
        os.environ.pop(variable, None)
    monkeypatch.setattr(settings, "hay_terminal", lambda: False)
    yield
    os.environ.clear()
    os.environ.update(original)
