import os
from getpass import getpass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env.local")

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass("🔑 OPENAI_API_KEY: ").strip()

if not os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = getpass("🔑 ANTHROPIC_API_KEY: ").strip()

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass(
        "🔑 Ingresá tu GOOGLE_API_KEY (gratis en aistudio.google.com/apikey): "
    ).strip()
