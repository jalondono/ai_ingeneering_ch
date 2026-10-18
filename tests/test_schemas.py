"""Validación de Esquemas con Pydantic: integridad y rango de los parámetros."""

import pytest
from pydantic import SecretStr, ValidationError

from backend.schemas import ChatMessage, LLMConfig, ModelResponse, Provider


def config(**cambios) -> LLMConfig:
    datos = {"provider": "openai", "model": "gpt-4o-mini"} | cambios
    return LLMConfig(**datos)


@pytest.mark.parametrize("temperatura", [0, 0.7, 2])
def test_temperatura_dentro_del_rango(temperatura):
    assert config(temperature=temperatura).temperature == temperatura


@pytest.mark.parametrize("temperatura", [-0.1, 2.1, 5])
def test_temperatura_fuera_del_rango_se_rechaza(temperatura):
    with pytest.raises(ValidationError, match="temperature"):
        config(temperature=temperatura)


@pytest.mark.parametrize("max_tokens", [0, -10])
def test_max_tokens_debe_ser_positivo(max_tokens):
    with pytest.raises(ValidationError, match="max_tokens"):
        config(max_tokens=max_tokens)


def test_valores_por_defecto():
    c = config()
    assert c.temperature == 0.7
    assert c.max_tokens == 1024


def test_convierte_strings_del_entorno_a_numeros():
    c = config(temperature="1.5", max_tokens="300")
    assert c.temperature == 1.5
    assert c.max_tokens == 300


def test_provider_invalido_se_rechaza():
    with pytest.raises(ValidationError, match="provider"):
        config(provider="mistral")


def test_api_key_no_se_muestra_al_imprimir():
    c = config(openai_api_key=SecretStr("sk-secreta"))
    assert "sk-secreta" not in repr(c)
    assert c.openai_api_key.get_secret_value() == "sk-secreta"


@pytest.mark.parametrize("rol", ["user", "assistant", "system"])
def test_chat_message_acepta_roles_validos(rol):
    assert ChatMessage(role=rol, content="hola").role == rol


def test_chat_message_rechaza_rol_invalido():
    with pytest.raises(ValidationError, match="role"):
        ChatMessage(role="admin", content="hola")


def test_chat_message_exige_content():
    with pytest.raises(ValidationError, match="content"):
        ChatMessage(role="user")


def test_model_response_estructura_el_error():
    r = ModelResponse(
        provider=Provider.OPENAI, model="gpt-4o-mini", content="", error="boom"
    )
    assert r.error == "boom"
    assert r.fallback_reason is None
