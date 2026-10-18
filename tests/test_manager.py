"""AsyncLLMManager: elige el proveedor por configuración y no deja escapar errores."""

import pytest
from pydantic import SecretStr, ValidationError

from backend.clients.anthropic import AnthropicClient
from backend.clients.gemini import GeminiClient
from backend.clients.openai import OpenAIClient
from backend.llm_manager import AsyncLLMManager
from backend.schemas import ChatMessage, LLMConfig, Provider
from tests.fakes import ClienteFalso

MENSAJES = [ChatMessage(role="user", content="¿Qué es la entropía?")]

CONFIG_OPENAI = LLMConfig(
    provider=Provider.OPENAI, model="gpt-4o-mini", openai_api_key=SecretStr("sk-test")
)
CONFIG_ANTHROPIC = LLMConfig(
    provider=Provider.ANTHROPIC,
    model="claude-sonnet-4-6",
    anthropic_api_key=SecretStr("sk-ant-test"),
)


def manager_falso(principal, respaldo=None) -> AsyncLLMManager:
    """Manager OpenAI -> Anthropic con los clientes reales reemplazados por dobles."""
    manager = AsyncLLMManager(
        CONFIG_OPENAI, fallback=CONFIG_ANTHROPIC if respaldo else None
    )
    manager._client = principal
    manager._fallback_client = respaldo
    return manager


async def juntar(stream) -> list[str]:
    return [chunk async for chunk in stream]


# --- Selección de proveedor por configuración ---------------------------


@pytest.mark.parametrize(
    ("provider", "campo_key", "clase"),
    [
        (Provider.OPENAI, "openai_api_key", OpenAIClient),
        (Provider.ANTHROPIC, "anthropic_api_key", AnthropicClient),
        (Provider.GEMINI, "google_api_key", GeminiClient),
    ],
)
def test_crea_el_cliente_segun_el_provider(provider, campo_key, clase):
    config = LLMConfig(provider=provider, model="m", **{campo_key: SecretStr("k")})
    assert isinstance(AsyncLLMManager(config)._client, clase)


def test_falta_la_api_key_del_provider():
    config = LLMConfig(provider=Provider.ANTHROPIC, model="claude-sonnet-4-6")
    with pytest.raises(ValueError, match="anthropic_api_key"):
        AsyncLLMManager(config)


@pytest.mark.parametrize(
    ("llm_provider", "clase", "modelo"),
    [
        ("openai", OpenAIClient, "gpt-4o-mini"),
        ("anthropic", AnthropicClient, "claude-sonnet-4-6"),
    ],
)
def test_desde_entorno_usa_llm_provider(monkeypatch, llm_provider, clase, modelo):
    monkeypatch.setenv("LLM_PROVIDER", llm_provider)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    manager = AsyncLLMManager.desde_entorno()

    assert isinstance(manager._client, clase)
    assert manager.config.model == modelo
    assert manager._fallback_client is None


def test_desde_entorno_agrega_el_fallback(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    manager = AsyncLLMManager.desde_entorno(Provider.OPENAI)

    assert isinstance(manager._client, OpenAIClient)
    assert isinstance(manager._fallback_client, AnthropicClient)


def test_desde_entorno_valida_los_parametros(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MAX_TOKENS", "0")
    with pytest.raises(ValidationError, match="max_tokens"):
        AsyncLLMManager.desde_entorno(Provider.OPENAI)


# --- Modo normal ----------------------------------------------------------


async def test_generate_devuelve_la_respuesta_del_principal():
    principal = ClienteFalso(Provider.OPENAI, texto="Mide el desorden.")
    resultado = await manager_falso(principal).generate(MENSAJES)
    assert resultado.content == "Mide el desorden."
    assert principal.recibidos == [MENSAJES]


async def test_generate_sin_fallback_devuelve_el_error_controlado():
    principal = ClienteFalso(Provider.OPENAI, error="Límite de cuota excedido")
    resultado = await manager_falso(principal).generate(MENSAJES)
    assert resultado.error == "Límite de cuota excedido"
    assert resultado.content == ""


async def test_generate_con_fallback_responde_el_respaldo():
    principal = ClienteFalso(Provider.OPENAI, error="Error de conexión")
    respaldo = ClienteFalso(Provider.ANTHROPIC, texto="Mide el desorden.")

    resultado = await manager_falso(principal, respaldo).generate(MENSAJES)

    assert resultado.provider == Provider.ANTHROPIC
    assert resultado.content == "Mide el desorden."
    assert resultado.fallback_reason == "openai falló: Error de conexión"


# --- Streaming ------------------------------------------------------------


async def test_stream_emite_los_tokens_del_principal():
    principal = ClienteFalso(Provider.OPENAI, tokens=("Mide ", "el ", "desorden."))
    tokens = await juntar(manager_falso(principal).generate_stream(MENSAJES))
    assert tokens == ["Mide ", "el ", "desorden."]


async def test_stream_sin_fallback_termina_con_un_aviso_en_vez_de_crash():
    principal = ClienteFalso(
        Provider.OPENAI, tokens=("Mide ",), error_stream="Límite de cuota excedido"
    )
    tokens = await juntar(manager_falso(principal).generate_stream(MENSAJES))
    assert tokens == ["Mide ", "\n[⚠️ Límite de cuota excedido]"]


async def test_stream_que_falla_antes_de_empezar_sigue_con_el_respaldo():
    principal = ClienteFalso(Provider.OPENAI, error_stream="Error de conexión")
    respaldo = ClienteFalso(Provider.ANTHROPIC, tokens=("Mide ", "el desorden."))

    tokens = await juntar(manager_falso(principal, respaldo).generate_stream(MENSAJES))

    assert tokens == ["Mide ", "el desorden."]


async def test_stream_que_falla_a_mitad_avisa_y_sigue_con_el_respaldo():
    principal = ClienteFalso(
        Provider.OPENAI, tokens=("Mi",), error_stream="Error de conexión"
    )
    respaldo = ClienteFalso(Provider.ANTHROPIC, tokens=("Mide el desorden.",))

    tokens = await juntar(manager_falso(principal, respaldo).generate_stream(MENSAJES))

    assert tokens[0] == "Mi"
    assert "openai falló a mitad de la respuesta" in tokens[1]
    assert tokens[2:] == ["Mide el desorden."]
