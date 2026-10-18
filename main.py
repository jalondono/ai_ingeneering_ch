"""Script de prueba del AsyncLLMManager.

Por defecto le pregunta "¿Qué es la entropía?" a cada proveedor de LLM_PROVIDER
(OpenAI y Anthropic), primero en modo normal y después en streaming.
"""

import argparse
import asyncio
import sys

from pydantic import SecretStr

from backend import settings
from backend.llm_manager import AsyncLLMManager
from backend.schemas import ChatMessage, LLMConfig, ModelResponse, Provider

PREGUNTA = [ChatMessage(role="user", content="¿Qué es la entropía?")]
KEY_INVALIDA = SecretStr("sk-key-invalida-a-proposito")


def mostrar(resultado: ModelResponse) -> None:
    if resultado.fallback_reason:
        print(f"⚠️ {resultado.fallback_reason}")
    texto = resultado.content if not resultado.error else f"❌ {resultado.error}"
    print(f"[{resultado.provider.value}] {texto}")


async def imprimir_stream(manager: AsyncLLMManager) -> None:
    async for chunk in manager.generate_stream(PREGUNTA):
        print(chunk, end="", flush=True)
    print()


async def prueba_basica() -> None:
    """Modo normal y streaming con cada proveedor de LLM_PROVIDER."""
    for provider in settings.proveedores_configurados():
        try:
            manager = AsyncLLMManager.desde_entorno(provider)
        except ValueError as e:
            print(f"⏭️  Se omite {provider.value}: {e}\n")
            continue

        print(f"── {provider.value} · modo normal ──")
        mostrar(await manager.generate(PREGUNTA))

        print(f"\n── {provider.value} · streaming ──")
        await imprimir_stream(manager)
        print()


async def prueba_resiliencia() -> None:
    """OpenAI con una API key inválida a propósito y sin fallback: no hay crash."""
    config_rota = LLMConfig(
        provider=Provider.OPENAI, model="gpt-4o-mini", openai_api_key=KEY_INVALIDA
    )
    manager = AsyncLLMManager(config_rota)

    resultado = await manager.generate(PREGUNTA)
    print("── modo normal ──")
    print("Error capturado (sin crash):", resultado.error)

    print("\n── streaming ──")
    await imprimir_stream(manager)
    print("\n¿El programa siguió vivo?: ✅ Sí")


async def prueba_fallback() -> None:
    """OpenAI falla a propósito y responde el proveedor de respaldo.

    El respaldo es LLM_FALLBACK_PROVIDER, o Anthropic si no está definido.
    """
    respaldo = settings.proveedor_fallback() or Provider.ANTHROPIC
    try:
        config_respaldo = settings.config_desde_entorno(respaldo)
    except ValueError as e:
        print(f"⏭️  No se puede probar el fallback con {respaldo.value}: {e}")
        return

    config_rota = LLMConfig(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        openai_api_key=KEY_INVALIDA,
        max_tokens=config_respaldo.max_tokens,
    )
    manager = AsyncLLMManager(config_rota, fallback=config_respaldo)

    print("── modo normal ──")
    mostrar(await manager.generate(PREGUNTA))

    print("\n── streaming ──")
    await imprimir_stream(manager)


PRUEBAS = {
    "basica": prueba_basica,
    "resiliencia": prueba_resiliencia,
    "fallback": prueba_fallback,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Pruebas del AsyncLLMManager")
    parser.add_argument(
        "prueba",
        nargs="?",
        default="basica",
        choices=PRUEBAS,
        help="prueba a ejecutar (por defecto: basica, modo normal y streaming)",
    )
    args = parser.parse_args()

    settings.cargar_entorno()
    try:
        asyncio.run(PRUEBAS[args.prueba]())
    except settings.ConfiguracionError as e:
        print(f"❌ Configuración inválida: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
