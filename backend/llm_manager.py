import logging
from collections.abc import AsyncGenerator
from typing import Self

from backend import settings
from backend.base import BaseLLMClient, LLMError
from backend.clients.anthropic import AnthropicClient
from backend.clients.gemini import GeminiClient
from backend.clients.openai import OpenAIClient
from backend.schemas import ChatMessage, LLMConfig, ModelResponse, Provider

logger = logging.getLogger(__name__)


class AsyncLLMManager:
    def __init__(self, config: LLMConfig, fallback: LLMConfig | None = None):
        """`fallback` es la configuración a usar si el proveedor principal falla."""
        self.config = config
        self.fallback = fallback
        self._client: BaseLLMClient = self._crear_cliente(config)
        self._fallback_client: BaseLLMClient | None = (
            self._crear_cliente(fallback) if fallback else None
        )

    @classmethod
    def desde_entorno(cls, provider: Provider | None = None) -> Self:
        """Arma el manager con la configuración del entorno (ver .env.example).

        Sin `provider` usa el primero de LLM_PROVIDER, así que cambiar de OpenAI a
        Anthropic es cambiar esa variable. Si LLM_FALLBACK_PROVIDER está definido,
        se agrega como fallback.
        """
        provider = provider or settings.proveedores_configurados()[0]
        fallback = settings.proveedor_fallback()
        return cls(
            settings.config_desde_entorno(provider),
            fallback=(
                settings.config_desde_entorno(fallback)
                if fallback and fallback != provider
                else None
            ),
        )

    @staticmethod
    def _crear_cliente(config: LLMConfig) -> BaseLLMClient:
        if config.provider == Provider.OPENAI:
            if not config.openai_api_key:
                raise ValueError("Falta openai_api_key en la configuración")
            return OpenAIClient(
                api_key=config.openai_api_key.get_secret_value(),
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

        if config.provider == Provider.ANTHROPIC:
            if not config.anthropic_api_key:
                raise ValueError("Falta anthropic_api_key en la configuración")
            return AnthropicClient(
                api_key=config.anthropic_api_key.get_secret_value(),
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

        if config.provider == Provider.GEMINI:
            if not config.google_api_key:
                raise ValueError("Falta google_api_key en la configuración")
            return GeminiClient(
                api_key=config.google_api_key.get_secret_value(),
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

        raise ValueError(f"Proveedor no soportado: {config.provider}")

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        resultado = await self._client.generate(messages)
        if not resultado.error or self._fallback_client is None:
            return resultado

        motivo = f"{self.config.provider.value} falló: {resultado.error}"
        logger.warning(
            "%s falló; reintentando con %s",
            self.config.provider.value,
            self.fallback.provider.value,
        )
        resultado_fallback = await self._fallback_client.generate(messages)
        return resultado_fallback.model_copy(update={"fallback_reason": motivo})

    async def generate_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        emitido = False  # si ya salió texto, avisamos antes de cambiar de proveedor
        error_principal: LLMError | None = None
        try:
            async for chunk in self._client.generate_stream(messages):
                emitido = True
                yield chunk
        except LLMError as e:
            error_principal = e

        if error_principal is None:
            return
        if self._fallback_client is None:
            yield f"\n[⚠️ {error_principal}]"
            return

        logger.warning(
            "%s falló en streaming; reintentando con %s",
            self.config.provider.value,
            self.fallback.provider.value,
        )
        if emitido:
            yield (
                f"\n\n[⚠️ {self.config.provider.value} falló a mitad de la respuesta; "
                f"continuando con {self.fallback.provider.value}]\n"
            )
        try:
            async for chunk in self._fallback_client.generate_stream(messages):
                yield chunk
        except LLMError as e:
            yield f"\n[⚠️ {e}]"
