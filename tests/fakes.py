"""Dobles de prueba: respuestas y errores de los SDKs sin tocar la red."""

from types import SimpleNamespace

import httpx2  # el cliente HTTP que usan los SDKs de OpenAI y Anthropic

from backend.base import BaseLLMClient, LLMError
from backend.schemas import ChatMessage, ModelResponse, Provider

_REQUEST = httpx2.Request("POST", "https://api.test/v1")


def error_rate_limit(sdk):
    """`sdk` es el módulo openai o anthropic."""
    respuesta = httpx2.Response(429, request=_REQUEST)
    return sdk.RateLimitError("Rate limit exceeded", response=respuesta, body=None)


def error_conexion(sdk):
    return sdk.APIConnectionError(request=_REQUEST)


def error_autenticacion(sdk):
    respuesta = httpx2.Response(401, request=_REQUEST)
    return sdk.AuthenticationError("Invalid API key", response=respuesta, body=None)


async def iterar(items, error=None):
    """Generador asíncrono que emite `items` y opcionalmente falla al final."""
    for item in items:
        yield item
    if error:
        raise error


# --- OpenAI ---------------------------------------------------------------


def respuesta_openai(texto):
    mensaje = SimpleNamespace(content=texto)
    return SimpleNamespace(choices=[SimpleNamespace(message=mensaje)])


def chunk_openai(texto):
    delta = SimpleNamespace(content=texto)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def sdk_openai(create):
    """Reemplaza a AsyncOpenAI: solo expone chat.completions.create."""
    completions = SimpleNamespace(create=create)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


# --- Anthropic ------------------------------------------------------------


def respuesta_anthropic(texto):
    return SimpleNamespace(content=[SimpleNamespace(text=texto)])


class StreamAnthropic:
    """Imita el context manager asíncrono que devuelve messages.stream()."""

    def __init__(self, textos=(), error=None):
        self.textos = textos
        self.error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def text_stream(self):
        return iterar(self.textos, self.error)


def sdk_anthropic(create=None, stream=None):
    """Reemplaza a AsyncAnthropic: solo expone messages.create y messages.stream."""
    return SimpleNamespace(messages=SimpleNamespace(create=create, stream=stream))


# --- Cliente falso para probar el manager --------------------------------


class ClienteFalso(BaseLLMClient):
    def __init__(
        self,
        provider: Provider,
        texto: str = "",
        error: str | None = None,
        tokens: tuple[str, ...] = (),
        error_stream: str | None = None,
    ):
        self.provider = provider
        self.texto = texto
        self.error = error
        self.tokens = tokens
        self.error_stream = error_stream
        self.recibidos: list[list[ChatMessage]] = []

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        self.recibidos.append(messages)
        return ModelResponse(
            provider=self.provider,
            model="falso",
            content="" if self.error else self.texto,
            error=self.error,
        )

    async def generate_stream(self, messages: list[ChatMessage]):
        self.recibidos.append(messages)
        for token in self.tokens:
            yield token
        if self.error_stream:
            raise LLMError(self.provider, self.error_stream)
