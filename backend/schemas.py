from enum import Enum

from pydantic import BaseModel, Field, SecretStr, field_validator


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class ChatMessage(BaseModel):
    role: str = Field(description="'user', 'assistant' o 'system'")
    content: str

    @field_validator("role")
    @classmethod
    def rol_valido(cls, v: str) -> str:
        roles_permitidos = {"user", "assistant", "system"}
        if v not in roles_permitidos:
            raise ValueError(
                f"role debe ser uno de {roles_permitidos}, recibido: '{v}'"
            )
        return v


class LLMConfig(BaseModel):
    provider: Provider
    model: str
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, gt=0)


class ModelResponse(BaseModel):
    provider: Provider
    model: str
    content: str
    error: str | None = None
    fallback_reason: str | None = None  # motivo por el que respondió el fallback
