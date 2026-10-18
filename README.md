# Cliente de LLM asíncrono unificado (OpenAI + Anthropic)

Pre-entrega 1: un *Unified Async LLM Client* en Python 3.12. La clase
`AsyncLLMManager` pone a OpenAI y Anthropic (y opcionalmente Gemini) detrás de
la misma interfaz:

- `generate(messages)`: respuesta completa (modo normal).
- `generate_stream(messages)`: generador asíncrono que devuelve el texto a medida
  que llega de la API.

El proveedor se elige por configuración (`LLM_PROVIDER` en el `.env`). Todas las
llamadas son no bloqueantes (`AsyncOpenAI` / `AsyncAnthropic` con `await`). Los
parámetros se validan con Pydantic. Los errores de red, de API key o de límite
de cuota se devuelven como un error controlado, sin cortar el programa.

## Cómo cumple la consigna

| Requisito | Dónde está | Cómo se verifica |
| --- | --- | --- |
| **Abstracción y gestión de proveedores (AsyncLLMManager)**: carga OpenAI o Anthropic según una variable de configuración | [`backend/llm_manager.py`](backend/llm_manager.py): `AsyncLLMManager` arma el cliente según `LLMConfig.provider`; `AsyncLLMManager.desde_entorno()` lo arma según `LLM_PROVIDER`. Interfaz común en [`backend/base.py`](backend/base.py) (`BaseLLMClient`). | [`tests/test_manager.py`](tests/test_manager.py) |
| **Asincronía**: llamadas no bloqueantes | [`backend/clients/openai.py`](backend/clients/openai.py) usa `AsyncOpenAI` y [`backend/clients/anthropic.py`](backend/clients/anthropic.py) usa `AsyncAnthropic`, siempre con `await` | `test_las_llamadas_no_bloquean_el_event_loop` en [`tests/test_clientes.py`](tests/test_clientes.py): tres llamadas en paralelo tardan lo mismo que una |
| **Streaming asíncrono**: generador de tokens en tiempo real | `generate_stream` en cada cliente y en el manager: `async for` sobre el stream del SDK + `yield` de cada fragmento | Tests `*_stream_*` en [`tests/test_clientes.py`](tests/test_clientes.py) y [`tests/test_manager.py`](tests/test_manager.py) |
| **Validación con Pydantic**: integridad y rango de los parámetros | [`backend/schemas.py`](backend/schemas.py): `ChatMessage` (role válido, content obligatorio), `LLMConfig` (`temperature` de 0 a 2, `max_tokens` > 0, API keys como `SecretStr`), `ModelResponse` | [`tests/test_schemas.py`](tests/test_schemas.py) |
| **Errores de red y rate limiting** sin crash | Cada cliente captura `RateLimitError`, `APIConnectionError` y `APIError`: en modo normal devuelve `ModelResponse.error` y en streaming termina con un aviso. Fallback opcional a otro proveedor. | Tests `*_captura_errores_sin_crash` y `python main.py resiliencia` |
| **Script de validación** | [`main.py`](main.py): carga el `.env` y pregunta "¿Qué es la entropía?" en modo normal y en streaming a OpenAI y a Anthropic | [`tests/test_main.py`](tests/test_main.py) y las salidas de ejemplo de abajo |

## Requisitos

- Python 3.12
- Una API key de OpenAI y otra de Anthropic (para probar contra las APIs reales)

Los tests no usan la red ni necesitan API keys.

## Instalación

Con `venv` y `pip`:

```bash
python3.12 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

O con [uv](https://docs.astral.sh/uv/), que respeta `.python-version` y `uv.lock`:

```bash
uv sync
```

## Variables de entorno

Copiá la plantilla y completá las API keys:

```bash
cp .env.example .env
```

| Variable | Obligatoria | Para qué sirve |
| --- | --- | --- |
| `LLM_PROVIDER` | No (por defecto `openai,anthropic`) | Proveedor que carga el `AsyncLLMManager`: `openai` o `anthropic` (también `gemini`). `main.py` prueba cada proveedor de la lista, separados por coma. |
| `OPENAI_API_KEY` | Sí, si usás OpenAI | API key de OpenAI |
| `ANTHROPIC_API_KEY` | Sí, si usás Anthropic | API key de Anthropic |
| `LLM_FALLBACK_PROVIDER` | No | Proveedor de respaldo si el principal falla. Vacío = sin fallback. |
| `GOOGLE_API_KEY` | Solo si usás Gemini | API key de Gemini |
| `LLM_TEMPERATURE` | No (por defecto `0.7`) | Temperatura, de 0 a 2 (validada con Pydantic) |
| `LLM_MAX_TOKENS` | No (por defecto `1024`; el `.env.example` usa `300`) | Máximo de tokens de la respuesta, mayor a 0 |
| `OPENAI_MODEL`, `ANTHROPIC_MODEL`, `GEMINI_MODEL` | No | Modelo de cada proveedor (por defecto `gpt-4o-mini`, `claude-sonnet-4-6`, `gemini-3.5-flash-lite`) |

`.env` está en `.gitignore`. Si falta la key de un proveedor y estás en una
terminal interactiva, el programa te la pide sin mostrarla. Si no hay terminal
(un CI o un evaluador automático), no se queda esperando: avisa que se omite ese
proveedor y sigue.

## Ejecutar el script de prueba

Desde la raíz del proyecto (con el entorno virtual activado, o anteponiendo
`uv run`):

```bash
python main.py
```

Le pregunta "¿Qué es la entropía?" a cada proveedor de `LLM_PROVIDER`, primero en
modo normal y después en streaming. Para probar un solo proveedor, cambiá la
variable:

```bash
LLM_PROVIDER=anthropic python main.py
```

Hay dos pruebas más:

```bash
python main.py resiliencia
```

Llama a OpenAI con una API key inválida a propósito y sin fallback: muestra el
error capturado en modo normal y en streaming, y el programa sigue vivo. No
necesita keys.

```bash
python main.py fallback
```

Llama a OpenAI con una API key inválida a propósito y responde el proveedor de
respaldo: `LLM_FALLBACK_PROVIDER`, o Anthropic si no está definido. Necesita la
key de ese proveedor.

## Salida de ejemplo

Las respuestas largas de los modelos y de las APIs están recortadas con `[…]`.

### `python main.py` con la key de Anthropic (sin la de OpenAI)

Modo normal y streaming contra la API real de Anthropic (`claude-sonnet-4-6`,
`LLM_MAX_TOKENS=300`). Como faltaba `OPENAI_API_KEY`, OpenAI se omite sin cortar
el programa.

```
⏭️  Se omite openai: falta OPENAI_API_KEY

── anthropic · modo normal ──
[anthropic] # Entropía

La entropía es un concepto fundamental que aparece en varios campos, con matices distintos:

## Termodinámica
Es una medida del **desorden o dispersión de energía** en un sistema. La Segunda Ley de la Termodinámica establece que:
- La entropía de un sistema aislado **siempre tiende a aumentar**
- Los procesos naturales van espontáneamente hacia estados de mayor desorden
- Por ejemplo: el hielo se derrite, el perfume se dispersa en el aire
[…]

── anthropic · streaming ──
# La Entropía

La entropía es un concepto fundamental que aparece en varios campos, principalmente en **termodinámica** y **teoría de la información**.

---

## En Termodinámica

Es una medida del **desorden o dispersión de energía** en un sistema.
[…]
```

### `python main.py` sin API keys ni terminal

```
⏭️  Se omite openai: falta OPENAI_API_KEY

⏭️  Se omite anthropic: falta ANTHROPIC_API_KEY

```

### `python main.py resiliencia`

```
── modo normal ──
Error capturado (sin crash): Error de la API de OpenAI: Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-key-i***************sito. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'code': 'invalid_api_key', 'param': None}, 'status': 401}

── streaming ──

[⚠️ Error durante el streaming: Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-key-i***************sito. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'code': 'invalid_api_key', 'param': None}, 'status': 401}]

¿El programa siguió vivo?: ✅ Sí
```

### `LLM_FALLBACK_PROVIDER=gemini python main.py fallback`

```
openai falló; reintentando con gemini
openai falló en streaming; reintentando con gemini
── modo normal ──
⚠️ openai falló: Error de la API de OpenAI: Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-key-i***************sito. […]', 'type': 'invalid_request_error', 'code': 'invalid_api_key', 'param': None}, 'status': 401}
[gemini] La **entropía** es un concepto fundamental de la física (específicamente de la termodinámica) que, para decirlo de manera sencilla, mide **el grado de desorden, aleatoriedad o caos** que hay en un sistema. […]

── streaming ──
La **entropía** es un concepto fundamental de la física (específicamente de la termodinámica) que, para decirlo de manera sencilla, mide **el grado de desorden, aleatoriedad o caos** que hay en un sistema. […]
```

## Tests

```bash
pytest
```

Los tests reemplazan los SDKs por dobles (`tests/fakes.py`), así que corren sin
red y sin API keys. Cubren:

- `test_schemas.py`: rangos de `temperature` y `max_tokens`, roles válidos y
  que la API key no aparezca al imprimir la configuración.
- `test_clientes.py`: respuestas, streaming token a token, errores de rate limit,
  conexión y autenticación de OpenAI y Anthropic, y que las llamadas no bloqueen
  el event loop.
- `test_manager.py`: elección del cliente según `LLMConfig` y `LLM_PROVIDER`,
  fallback en modo normal y en streaming, y errores controlados.
- `test_settings.py`: lectura del `.env` y que sin terminal no se pida nada por
  teclado.
- `test_main.py`: `main.py` hace la pregunta de la consigna en los dos modos, y
  sin keys ni terminal termina bien en vez de colgarse.

Salida (sin las líneas de configuración de pytest):

```
============================= test session starts ==============================
platform darwin -- Python 3.12.3, pytest-9.1.1, pluggy-1.7.0
configfile: pyproject.toml
testpaths: tests
plugins: asyncio-1.4.0, anyio-4.15.1
collected 65 items

tests/test_clientes.py ...............                                   [ 23%]
tests/test_main.py ...                                                   [ 27%]
tests/test_manager.py ...............                                    [ 50%]
tests/test_schemas.py ..................                                 [ 78%]
tests/test_settings.py ..............                                    [100%]

============================== 65 passed in 8.32s ==============================
```

## Usarlo desde tu código

```python
import asyncio

from backend import settings
from backend.llm_manager import AsyncLLMManager
from backend.schemas import ChatMessage


async def main():
    settings.cargar_entorno()  # lee el .env
    manager = AsyncLLMManager.desde_entorno()  # OpenAI o Anthropic según LLM_PROVIDER
    pregunta = [ChatMessage(role="user", content="¿Qué es la entropía?")]

    resultado = await manager.generate(pregunta)
    print(resultado.error or resultado.content)

    async for fragmento in manager.generate_stream(pregunta):
        print(fragmento, end="", flush=True)


asyncio.run(main())
```

También se puede armar la configuración a mano:

```python
from pydantic import SecretStr

from backend.schemas import LLMConfig, Provider

config = LLMConfig(
    provider=Provider.ANTHROPIC,
    model="claude-sonnet-4-6",
    anthropic_api_key=SecretStr("sk-ant-..."),
    temperature=0.7,
    max_tokens=300,
)
manager = AsyncLLMManager(config)
```

Con `fallback`, el manager usa otro proveedor cuando el principal falla:

- En modo normal, `resultado.provider` dice quién respondió y
  `resultado.fallback_reason` guarda el error original.
- En streaming, si el principal falla antes de emitir texto, sigue el respaldo
  sin avisar. Si falla a mitad de la respuesta, emite una línea de aviso y el
  respaldo responde desde el principio.

## Estructura del proyecto

```
main.py               # script de prueba: modo normal y streaming
backend/
├── schemas.py        # Pydantic: Provider, ChatMessage, LLMConfig, ModelResponse
├── base.py           # BaseLLMClient: interfaz común de todos los clientes
├── llm_manager.py    # AsyncLLMManager: elige el cliente según la configuración
├── settings.py       # lee el .env y arma el LLMConfig de cada proveedor
└── clients/
    ├── openai.py     # AsyncOpenAI
    ├── anthropic.py  # AsyncAnthropic
    └── gemini.py     # google-genai (opcional, como fallback)
tests/                # pytest con dobles de los SDKs
.env.example          # plantilla de variables de entorno
requirements.txt      # dependencias para pip
```

## Desarrollo

Lint y formato con [ruff](https://docs.astral.sh/ruff/), que corre en cada commit
con pre-commit:

```bash
uv run pre-commit install
```

```bash
uv run ruff check --fix .
```

```bash
uv run ruff format .
```
