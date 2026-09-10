"""LLM configuration for the agents."""

import os
from typing import Any, Mapping

from dotenv import load_dotenv
import httpx
from pydantic import BaseModel


class _StructuredOpenRouterModel:
    def __init__(self, model: "OpenRouterChatModel", schema: type[BaseModel]):
        self.model = model
        self.schema = schema

    def invoke(self, messages: Any) -> BaseModel:
        content = self.model._request(
            messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": self.schema.__name__,
                    "strict": True,
                    "schema": self.model._strict_json_schema(self.schema.model_json_schema()),
                },
            },
        )
        return self.schema.model_validate_json(content)


class OpenRouterChatModel:
    """Minimal synchronous OpenRouter adapter used by Nahaj's agents."""

    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, *, api_key: str, model: str, temperature: float, timeout: int, max_retries: int):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def with_structured_output(self, schema: type[BaseModel]) -> _StructuredOpenRouterModel:
        return _StructuredOpenRouterModel(self, schema)

    def invoke(self, messages: Any) -> dict[str, str]:
        return {"content": self._request(messages)}

    def _request(self, messages: Any, response_format: Mapping[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(messages),
            "temperature": self.temperature,
        }
        if response_format is not None:
            payload["response_format"] = dict(response_format)

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                data = response.json()
                if response.is_error:
                    detail = data.get("error", {}) if isinstance(data, Mapping) else {}
                    message = detail.get("message") if isinstance(detail, Mapping) else None
                    raise RuntimeError(f"OpenRouter returned HTTP {response.status_code}: {message or 'request failed'}")
                content = data["choices"][0]["message"].get("content", "")
                if isinstance(content, list):
                    content = "".join(
                        str(item.get("text", "")) if isinstance(item, Mapping) else str(item)
                        for item in content
                    )
                if not str(content).strip():
                    raise RuntimeError("OpenRouter returned an empty response.")
                return str(content).strip()
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
        raise RuntimeError(str(last_error or "OpenRouter request failed."))

    @classmethod
    def _strict_json_schema(cls, value: Any) -> Any:
        if isinstance(value, dict):
            result = {key: cls._strict_json_schema(item) for key, item in value.items()}
            properties = result.get("properties")
            if isinstance(properties, dict):
                result["required"] = list(properties)
                result["additionalProperties"] = False
            return result
        if isinstance(value, list):
            return [cls._strict_json_schema(item) for item in value]
        return value

    @staticmethod
    def _messages(messages: Any) -> list[dict[str, str]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        normalized: list[dict[str, str]] = []
        for message in list(messages or []):
            if isinstance(message, Mapping):
                role = str(message.get("role") or "user")
                content = message.get("content", "")
            else:
                role = str(getattr(message, "type", "user"))
                content = getattr(message, "content", "")
            normalized.append({"role": role, "content": str(content)})
        return normalized

 
class config:
    def __init__(self):
        load_dotenv()

        api_key = os.getenv("OPENROUTER_API_KEY")
        model = os.getenv("OPENROUTER_MODEL")
        if not api_key or not model:
            raise ValueError("OPENROUTER_API_KEY and OPENROUTER_MODEL are required.")

        self.OPENROUTER_API_KEY = api_key
        self.llm = OpenRouterChatModel(
            api_key=api_key,
            model=model,
            temperature=float(os.getenv("AGENT_TEMPERATURE", "0")),
            timeout=int(os.getenv("AGENT_TIMEOUT_SECONDS", "45")),
            max_retries=int(os.getenv("AGENT_MAX_RETRIES", "2")),
        )
