import argparse
import asyncio
import os

from pydantic import SecretStr

from backend import settings  # noqa: F401  (carga las API keys en os.environ)
from backend.llm_manager import AsyncLLMManager
from backend.schemas import ChatMessage, LLMConfig, Provider

pregunta = [
    ChatMessage(role="user", content="Explicá en una sola oración qué es un LLM.")
]


def crear_config_gemini() -> LLMConfig:
    """Gemini es el fallback: responde si OpenAI o Anthropic fallan."""
    return LLMConfig(
        provider=Provider.GEMINI,
        model="gemini-3.5-flash-lite",
        google_api_key=SecretStr(os.environ["GOOGLE_API_KEY"]),
        temperature=0.7,
        max_tokens=200,
    )


def crear_manager_openai() -> AsyncLLMManager:
    config_openai = LLMConfig(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        openai_api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
        temperature=0.7,
        max_tokens=200,
    )
    return AsyncLLMManager(config_openai, fallback=crear_config_gemini())


def crear_manager_anthropic() -> AsyncLLMManager:
    config_anthropic = LLMConfig(
        provider=Provider.ANTHROPIC,
        model="claude-sonnet-4-6",
        anthropic_api_key=SecretStr(os.environ["ANTHROPIC_API_KEY"]),
        temperature=0.7,
        max_tokens=200,
    )
    return AsyncLLMManager(config_anthropic, fallback=crear_config_gemini())


def mostrar(etiqueta: str, resultado) -> None:
    if resultado.fallback_reason:
        print(f"⚠️ {resultado.fallback_reason}")
    texto = resultado.content if not resultado.error else f"❌ {resultado.error}"
    print(f"{etiqueta} ({resultado.provider.value}):", texto)


async def prueba_normal() -> None:
    """Prueba en modo normal (Anthropic)."""
    manager_anthropic = crear_manager_anthropic()

    resultado = await manager_anthropic.generate(pregunta)
    mostrar("🟣 Anthropic", resultado)


async def prueba_streaming() -> None:
    """Prueba en modo streaming (los dos proveedores)."""
    manager_openai = crear_manager_openai()
    manager_anthropic = crear_manager_anthropic()

    print("🟢 Streaming OpenAI:")
    async for chunk in manager_openai.generate_stream(pregunta):
        print(chunk, end="", flush=True)

    print("\n\n🟣 Streaming Anthropic:")
    async for chunk in manager_anthropic.generate_stream(pregunta):
        print(chunk, end="", flush=True)
    print()


async def prueba_resiliencia() -> None:
    """Prueba de resiliencia (API key inválida a propósito, sin fallback)."""
    config_rota = LLMConfig(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        openai_api_key=SecretStr("sk-key-invalida-a-proposito"),
    )

    manager_roto = AsyncLLMManager(config_rota)
    resultado = await manager_roto.generate(pregunta)

    print("¿El programa siguió vivo?: ✅ Sí")
    print("Error capturado (sin crash):", resultado.error)


async def prueba_fallback() -> None:
    """Prueba de fallback: OpenAI falla a propósito y responde Gemini."""
    config_rota = LLMConfig(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        openai_api_key=SecretStr("sk-key-invalida-a-proposito"),
        max_tokens=200,
    )

    manager = AsyncLLMManager(config_rota, fallback=crear_config_gemini())

    resultado = await manager.generate(pregunta)
    mostrar("🔵 Respuesta", resultado)

    print("\n🔵 Streaming con fallback:")
    async for chunk in manager.generate_stream(pregunta):
        print(chunk, end="", flush=True)
    print()


PRUEBAS = {
    "normal": prueba_normal,
    "streaming": prueba_streaming,
    "resiliencia": prueba_resiliencia,
    "fallback": prueba_fallback,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pruebas del AsyncLLMManager")
    parser.add_argument("prueba", choices=PRUEBAS, help="prueba a ejecutar")
    args = parser.parse_args()

    asyncio.run(PRUEBAS[args.prueba]())


if __name__ == "__main__":
    main()
