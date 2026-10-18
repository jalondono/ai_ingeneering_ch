"""Configuración leída del entorno (.env), sin bloquear el programa si falta algo.

Importar este módulo no tiene efectos: el .env se carga al llamar a cargar_entorno()
y las API keys se piden por terminal solo si hay una terminal interactiva.
"""

import os
import sys
from getpass import getpass
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr

from backend.schemas import LLMConfig, Provider

RAIZ = Path(__file__).resolve().parent.parent

# variable de entorno con la API key de cada proveedor; en minúsculas es el campo
# correspondiente de LLMConfig (OPENAI_API_KEY -> openai_api_key)
API_KEYS = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.GEMINI: "GOOGLE_API_KEY",
}

# se pueden cambiar con OPENAI_MODEL, ANTHROPIC_MODEL y GEMINI_MODEL
MODELOS_POR_DEFECTO = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-sonnet-4-6",
    Provider.GEMINI: "gemini-3.5-flash-lite",
}

PROVEEDORES_POR_DEFECTO = "openai,anthropic"


class ConfiguracionError(ValueError):
    """Falta una variable de entorno o tiene un valor inválido."""


def cargar_entorno(ruta: Path = RAIZ / ".env") -> None:
    """Carga el .env sin pisar las variables que ya estén definidas."""
    load_dotenv(ruta)


def hay_terminal() -> bool:
    return sys.stdin is not None and sys.stdin.isatty()


def obtener_api_key(provider: Provider) -> SecretStr:
    """Devuelve la API key del proveedor.

    Si no está en el entorno, la pide por terminal solo cuando hay una terminal
    interactiva; en un entorno sin terminal (CI, evaluador automático) lanza
    ConfiguracionError en vez de quedarse esperando.
    """
    nombre = API_KEYS[provider]
    valor = os.environ.get(nombre, "").strip()
    if not valor and hay_terminal():
        valor = getpass(f"🔑 {nombre}: ").strip()
        os.environ[nombre] = valor  # para no volver a preguntarla en esta ejecución
    if not valor:
        raise ConfiguracionError(f"falta {nombre}")
    return SecretStr(valor)


def _parsear_proveedor(valor: str, variable: str) -> Provider:
    try:
        return Provider(valor.strip().lower())
    except ValueError:
        opciones = ", ".join(p.value for p in Provider)
        raise ConfiguracionError(
            f"{variable} tiene un proveedor inválido: '{valor.strip()}' "
            f"(opciones: {opciones})"
        ) from None


def proveedores_configurados() -> list[Provider]:
    """Proveedores de LLM_PROVIDER: uno solo o varios separados por coma."""
    valor = os.environ.get("LLM_PROVIDER", "").strip() or PROVEEDORES_POR_DEFECTO
    proveedores = [
        _parsear_proveedor(p, "LLM_PROVIDER") for p in valor.split(",") if p.strip()
    ]
    if not proveedores:
        raise ConfiguracionError("LLM_PROVIDER no tiene ningún proveedor")
    return proveedores


def proveedor_fallback() -> Provider | None:
    """Proveedor de LLM_FALLBACK_PROVIDER, o None si no está definido."""
    valor = os.environ.get("LLM_FALLBACK_PROVIDER", "").strip()
    return _parsear_proveedor(valor, "LLM_FALLBACK_PROVIDER") if valor else None


def config_desde_entorno(provider: Provider) -> LLMConfig:
    """Arma el LLMConfig del proveedor con las variables de entorno.

    Pydantic valida los valores: un LLM_TEMPERATURE fuera de 0..2 o un
    LLM_MAX_TOKENS menor a 1 lanza ValidationError.
    """
    extras = {}
    if os.environ.get("LLM_TEMPERATURE"):
        extras["temperature"] = os.environ["LLM_TEMPERATURE"]
    if os.environ.get("LLM_MAX_TOKENS"):
        extras["max_tokens"] = os.environ["LLM_MAX_TOKENS"]

    modelo = os.environ.get(f"{provider.value.upper()}_MODEL", "").strip()
    return LLMConfig(
        provider=provider,
        model=modelo or MODELOS_POR_DEFECTO[provider],
        **{API_KEYS[provider].lower(): obtener_api_key(provider)},
        **extras,
    )
