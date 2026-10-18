"""Clientes de OpenAI y Anthropic con el SDK reemplazado por dobles (sin red)."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import anthropic
import openai
import pytest

from backend.base import LLMError
from backend.clients.anthropic import AnthropicClient
from backend.clients.openai import OpenAIClient
from backend.schemas import ChatMessage, Provider
from tests import fakes

MENSAJES = [ChatMessage(role="user", content="¿Qué es la entropía?")]
MENSAJES_SDK = [{"role": "user", "content": "¿Qué es la entropía?"}]


def cliente_openai(create) -> OpenAIClient:
    cliente = OpenAIClient(
        api_key="sk-test", model="gpt-4o-mini", temperature=0.5, max_tokens=50
    )
    cliente._client = fakes.sdk_openai(create)
    return cliente


def cliente_anthropic(create=None, stream=None) -> AnthropicClient:
    cliente = AnthropicClient(
        api_key="sk-ant-test", model="claude-sonnet-4-6", temperature=0.5, max_tokens=50
    )
    cliente._client = fakes.sdk_anthropic(create=create, stream=stream)
    return cliente


async def juntar(stream) -> list[str]:
    return [chunk async for chunk in stream]


# --- OpenAI ---------------------------------------------------------------


async def test_openai_generate_devuelve_model_response():
    create = AsyncMock(return_value=fakes.respuesta_openai("Mide el desorden."))

    resultado = await cliente_openai(create).generate(MENSAJES)

    assert resultado.provider == Provider.OPENAI
    assert resultado.content == "Mide el desorden."
    assert resultado.error is None
    create.assert_awaited_once_with(
        model="gpt-4o-mini", messages=MENSAJES_SDK, temperature=0.5, max_tokens=50
    )


async def test_openai_stream_emite_los_tokens_a_medida_que_llegan():
    chunks = [fakes.chunk_openai(t) for t in ("Mide ", None, "el ", "desorden.")]
    create = AsyncMock(return_value=fakes.iterar(chunks))

    tokens = await juntar(cliente_openai(create).generate_stream(MENSAJES))

    assert tokens == ["Mide ", "el ", "desorden."]  # los chunks vacíos se saltean
    assert create.await_args.kwargs["stream"] is True


@pytest.mark.parametrize(
    ("error", "mensaje"),
    [
        (fakes.error_rate_limit(openai), "Límite de cuota excedido"),
        (fakes.error_conexion(openai), "Error de conexión"),
        (fakes.error_autenticacion(openai), "Error de la API de OpenAI"),
    ],
)
async def test_openai_generate_captura_errores_sin_crash(error, mensaje):
    resultado = await cliente_openai(AsyncMock(side_effect=error)).generate(MENSAJES)

    assert resultado.error.startswith(mensaje)
    assert resultado.content == ""
    assert resultado.provider == Provider.OPENAI


@pytest.mark.parametrize(
    "error", [fakes.error_rate_limit(openai), fakes.error_conexion(openai)]
)
async def test_openai_stream_convierte_errores_en_llm_error(error):
    stream = fakes.iterar([fakes.chunk_openai("Mide ")], error=error)
    cliente = cliente_openai(AsyncMock(return_value=stream))

    with pytest.raises(LLMError) as info:
        await juntar(cliente.generate_stream(MENSAJES))

    assert info.value.provider == Provider.OPENAI


# --- Anthropic ------------------------------------------------------------


async def test_anthropic_generate_devuelve_model_response():
    create = AsyncMock(return_value=fakes.respuesta_anthropic("Mide el desorden."))

    resultado = await cliente_anthropic(create=create).generate(MENSAJES)

    assert resultado.provider == Provider.ANTHROPIC
    assert resultado.content == "Mide el desorden."
    assert resultado.error is None
    create.assert_awaited_once_with(
        model="claude-sonnet-4-6",
        max_tokens=50,
        extra_body={"temperature": 0.5},
        messages=MENSAJES_SDK,
    )


async def test_anthropic_stream_emite_los_tokens_a_medida_que_llegan():
    stream = MagicMock(return_value=fakes.StreamAnthropic(["Mide ", "el desorden."]))

    tokens = await juntar(cliente_anthropic(stream=stream).generate_stream(MENSAJES))

    assert tokens == ["Mide ", "el desorden."]
    assert stream.call_args.kwargs["messages"] == MENSAJES_SDK


@pytest.mark.parametrize(
    ("error", "mensaje"),
    [
        (fakes.error_rate_limit(anthropic), "Límite de cuota excedido"),
        (fakes.error_conexion(anthropic), "Error de conexión"),
        (fakes.error_autenticacion(anthropic), "Error de la API de Anthropic"),
    ],
)
async def test_anthropic_generate_captura_errores_sin_crash(error, mensaje):
    cliente = cliente_anthropic(create=AsyncMock(side_effect=error))

    resultado = await cliente.generate(MENSAJES)

    assert resultado.error.startswith(mensaje)
    assert resultado.content == ""
    assert resultado.provider == Provider.ANTHROPIC


async def test_anthropic_stream_convierte_errores_en_llm_error():
    error = fakes.error_rate_limit(anthropic)
    stream = MagicMock(return_value=fakes.StreamAnthropic(["Mide "], error=error))

    with pytest.raises(LLMError) as info:
        await juntar(cliente_anthropic(stream=stream).generate_stream(MENSAJES))

    assert info.value.provider == Provider.ANTHROPIC


# --- Asincronía -----------------------------------------------------------


@pytest.mark.parametrize("proveedor", ["openai", "anthropic"])
async def test_las_llamadas_no_bloquean_el_event_loop(proveedor):
    """Tres llamadas de 0.2 s en paralelo tardan ~0.2 s, no 0.6 s."""

    async def create_lento(**_kwargs):
        await asyncio.sleep(0.2)
        if proveedor == "openai":
            return fakes.respuesta_openai("ok")
        return fakes.respuesta_anthropic("ok")

    if proveedor == "openai":
        cliente = cliente_openai(create_lento)
    else:
        cliente = cliente_anthropic(create=create_lento)

    inicio = time.perf_counter()
    resultados = await asyncio.gather(*(cliente.generate(MENSAJES) for _ in range(3)))
    duracion = time.perf_counter() - inicio

    assert [r.content for r in resultados] == ["ok", "ok", "ok"]
    assert duracion < 0.45
