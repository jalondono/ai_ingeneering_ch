from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from backend.schemas import ChatMessage, ModelResponse, Provider


class LLMError(Exception):
    """Error controlado de un proveedor durante el streaming.

    Los clientes lo lanzan en vez de dejar escapar la excepción del SDK, para que
    el manager pueda decidir si continúa con el proveedor de fallback.
    """

    def __init__(self, provider: Provider, mensaje: str):
        super().__init__(mensaje)
        self.provider = provider


class BaseLLMClient(ABC):
    """Contrato que todo cliente de LLM debe cumplir, sin importar el proveedor real detrás."""

    @abstractmethod
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Genera una respuesta completa (modo normal, no streaming)."""
        raise NotImplementedError

    @abstractmethod
    async def generate_stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str]:
        """Genera la respuesta token a token (modo streaming).

        Ante un error del proveedor lanza LLMError en vez de cortar el stream.
        """
        raise NotImplementedError
        yield  # nunca se ejecuta; solo le indica a Python que este método es un generador
