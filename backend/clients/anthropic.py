from collections.abc import AsyncGenerator

from anthropic import (
    APIConnectionError as AnthropicConnectionError,
)
from anthropic import (
    APIError as AnthropicAPIError,
)
from anthropic import (
    AsyncAnthropic,
)
from anthropic import (
    RateLimitError as AnthropicRateLimitError,
)

from backend.base import BaseLLMClient, LLMError
from backend.schemas import ChatMessage, ModelResponse, Provider


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self._client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,  # obligatorio en Anthropic, a diferencia de OpenAI
                # el SDK 1.x ya no acepta temperature como argumento; va en el body
                extra_body={"temperature": self.temperature},
                messages=[m.model_dump() for m in messages],
            )
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=self.model,
                content=response.content[0].text,
            )
        except AnthropicRateLimitError as e:
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=self.model,
                content="",
                error=f"Límite de cuota excedido: {e}",
            )
        except AnthropicConnectionError as e:
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=self.model,
                content="",
                error=f"Error de conexión: {e}",
            )
        except AnthropicAPIError as e:
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=self.model,
                content="",
                error=f"Error de la API de Anthropic: {e}",
            )

    async def generate_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        try:
            async with self._client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                extra_body={"temperature": self.temperature},
                messages=[m.model_dump() for m in messages],
            ) as stream:
                async for texto in stream.text_stream:
                    yield texto
        except (
            AnthropicRateLimitError,
            AnthropicConnectionError,
            AnthropicAPIError,
        ) as e:
            raise LLMError(
                Provider.ANTHROPIC, f"Error durante el streaming: {e}"
            ) from e
