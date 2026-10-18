"""Configuración desde el entorno sin bloquear cuando no hay terminal."""

import os

import pytest
from pydantic import ValidationError

from backend import settings
from backend.schemas import Provider


def no_preguntar(_prompt):
    pytest.fail("getpass no debería llamarse sin terminal interactiva")


def test_sin_key_y_sin_terminal_falla_sin_bloquear(monkeypatch):
    monkeypatch.setattr(settings, "getpass", no_preguntar)
    with pytest.raises(settings.ConfiguracionError, match="falta OPENAI_API_KEY"):
        settings.obtener_api_key(Provider.OPENAI)


def test_key_vacia_cuenta_como_faltante(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    monkeypatch.setattr(settings, "getpass", no_preguntar)
    with pytest.raises(settings.ConfiguracionError, match="falta ANTHROPIC_API_KEY"):
        settings.obtener_api_key(Provider.ANTHROPIC)


def test_key_del_entorno_no_pregunta(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-del-entorno")
    monkeypatch.setattr(settings, "hay_terminal", lambda: True)
    monkeypatch.setattr(settings, "getpass", no_preguntar)
    key = settings.obtener_api_key(Provider.OPENAI)
    assert key.get_secret_value() == "sk-del-entorno"


def test_con_terminal_pide_la_key_que_falta(monkeypatch):
    monkeypatch.setattr(settings, "hay_terminal", lambda: True)
    monkeypatch.setattr(settings, "getpass", lambda _prompt: "  sk-tipeada ")
    key = settings.obtener_api_key(Provider.ANTHROPIC)
    assert key.get_secret_value() == "sk-tipeada"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-tipeada"


def test_proveedores_por_defecto_son_openai_y_anthropic():
    assert settings.proveedores_configurados() == [Provider.OPENAI, Provider.ANTHROPIC]


@pytest.mark.parametrize(
    ("valor", "esperado"),
    [
        ("anthropic", [Provider.ANTHROPIC]),
        ("openai", [Provider.OPENAI]),
        (" OpenAI , anthropic ", [Provider.OPENAI, Provider.ANTHROPIC]),
    ],
)
def test_llm_provider_elige_los_proveedores(monkeypatch, valor, esperado):
    monkeypatch.setenv("LLM_PROVIDER", valor)
    assert settings.proveedores_configurados() == esperado


def test_llm_provider_invalido(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    with pytest.raises(settings.ConfiguracionError, match="mistral"):
        settings.proveedores_configurados()


def test_fallback_es_opcional(monkeypatch):
    assert settings.proveedor_fallback() is None
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "gemini")
    assert settings.proveedor_fallback() == Provider.GEMINI


def test_config_desde_entorno_lee_modelo_y_parametros(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-otro")
    monkeypatch.setenv("LLM_TEMPERATURE", "1.5")
    monkeypatch.setenv("LLM_MAX_TOKENS", "100")

    config = settings.config_desde_entorno(Provider.OPENAI)

    assert config.provider == Provider.OPENAI
    assert config.model == "gpt-otro"
    assert config.temperature == 1.5
    assert config.max_tokens == 100
    assert config.openai_api_key.get_secret_value() == "sk-test"


def test_config_desde_entorno_usa_modelo_por_defecto(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    config = settings.config_desde_entorno(Provider.ANTHROPIC)
    assert config.model == "claude-sonnet-4-6"
    assert config.anthropic_api_key.get_secret_value() == "sk-ant-test"


def test_temperatura_invalida_en_el_entorno_la_rechaza_pydantic(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_TEMPERATURE", "3")
    with pytest.raises(ValidationError, match="temperature"):
        settings.config_desde_entorno(Provider.OPENAI)


def test_cargar_entorno_no_pisa_variables_existentes(monkeypatch, tmp_path):
    archivo = tmp_path / ".env"
    archivo.write_text("LLM_PROVIDER=anthropic\nOPENAI_API_KEY=del-archivo\n")
    monkeypatch.setenv("OPENAI_API_KEY", "del-entorno")

    settings.cargar_entorno(archivo)

    assert os.environ["LLM_PROVIDER"] == "anthropic"
    assert os.environ["OPENAI_API_KEY"] == "del-entorno"
