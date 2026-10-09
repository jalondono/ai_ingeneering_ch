from collections.abc import AsyncGenerator

from google import genai
from google.genai import types

from backend.base import BaseLLMClient, LLMError
from backend.schemas import ChatMessage, ModelResponse, Provider


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _convertir_mensajes(self, messages: list[ChatMessage]):
        """Gemini separa el system prompt del resto, y llama 'model' al rol del asistente."""
        contents = []
        system_instruction = None
        for m in messages:
            if m.role == "system":
                system_instruction = m.content
            else:
                rol_gemini = "model" if m.role == "assistant" else "user"
                contents.append(
                    types.Content(role=rol_gemini, parts=[types.Part(text=m.content)])
                )
        return contents, system_instruction

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        try:
            contents, system_instruction = self._convertir_mensajes(messages)
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    system_instruction=system_instruction,
                    # sin tools no hay function calling; esto evita un warning del SDK
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
            return ModelResponse(
                provider=Provider.GEMINI, model=self.model, content=response.text
            )
        except Exception as e:
            return ModelResponse(
                provider=Provider.GEMINI,
                model=self.model,
                content="",
                error=f"Error de la API de Gemini: {e}",
            )

    async def generate_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        try:
            contents, system_instruction = self._convertir_mensajes(messages)
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    system_instruction=system_instruction,
                    # sin tools no hay function calling; esto evita un warning del SDK
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise LLMError(Provider.GEMINI, f"Error durante el streaming: {e}") from e
