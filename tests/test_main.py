"""El script de prueba: modo normal y streaming, y que nunca se quede colgado."""

import os
import subprocess
import sys

import main
from backend.llm_manager import AsyncLLMManager
from backend.schemas import Provider
from backend.settings import RAIZ
from tests.fakes import ClienteFalso
from tests.test_manager import manager_falso


def correr_main(*args: str, **variables: str) -> subprocess.CompletedProcess:
    """Corre `python main.py` como lo haría un evaluador: sin terminal (stdin vacío).

    Las variables del proyecto van vacías para que un .env local no las complete
    y el test nunca llame a una API real.
    """
    entorno = (
        os.environ
        | {
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "LLM_PROVIDER": "",
            "LLM_FALLBACK_PROVIDER": "",
            "LLM_TEMPERATURE": "",
            "LLM_MAX_TOKENS": "",
        }
        | variables
    )
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=RAIZ,
        env=entorno,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_sin_keys_ni_terminal_no_se_bloquea_ni_rompe():
    resultado = correr_main()

    assert resultado.returncode == 0, resultado.stderr
    assert "Se omite openai: falta OPENAI_API_KEY" in resultado.stdout
    assert "Se omite anthropic: falta ANTHROPIC_API_KEY" in resultado.stdout


def test_llm_provider_invalido_sale_con_un_error_claro():
    resultado = correr_main(LLM_PROVIDER="mistral")

    assert resultado.returncode == 1
    assert "Configuración inválida" in resultado.stderr
    assert "Traceback" not in resultado.stderr


async def test_prueba_basica_pregunta_en_normal_y_streaming(monkeypatch, capsys):
    clientes = []

    def desde_entorno(provider=None):
        cliente = ClienteFalso(provider, texto="Mide el desorden.", tokens=("Mi", "de"))
        clientes.append(cliente)
        return manager_falso(cliente)

    monkeypatch.setattr(AsyncLLMManager, "desde_entorno", desde_entorno)

    await main.prueba_basica()

    salida = capsys.readouterr().out
    for proveedor in ("openai", "anthropic"):
        assert f"── {proveedor} · modo normal ──" in salida
        assert f"── {proveedor} · streaming ──" in salida
    assert [c.provider for c in clientes] == [Provider.OPENAI, Provider.ANTHROPIC]
    for cliente in clientes:
        # una llamada en modo normal y otra en streaming, con la pregunta de la consigna
        assert len(cliente.recibidos) == 2
        assert cliente.recibidos[0][0].content == "¿Qué es la entropía?"
