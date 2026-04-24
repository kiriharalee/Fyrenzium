"""OpenRouter client helpers for translation calls.

OpenRouter exposes an OpenAI-compatible chat completions endpoint. This module
keeps the implementation focused on request/response handling so translation
logic can live in the orchestration layer.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "minimax/minimax-m2.5:free"


class OpenRouterError(RuntimeError):
    """Base error for OpenRouter client failures."""


class OpenRouterAPIError(OpenRouterError):
    """Raised when OpenRouter returns a non-success response."""

    def __init__(self, status_code: int, message: str, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class OpenRouterRateLimitError(OpenRouterAPIError):
    """Raised when OpenRouter rate limits a request."""


@dataclass(frozen=True)
class ChatMessage:
    """Simple chat message structure compatible with OpenAI-style APIs."""

    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def _normalize_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        if not role:
            raise ValueError("each message requires a role")
        content = message.get("content", "")
        if isinstance(content, list):
            normalized.append({"role": role, "content": content})
        else:
            normalized.append({"role": role, "content": str(content)})
    return normalized


class OpenRouterClient:
    """Small OpenRouter API client for translation and other chat tasks."""

    def __init__(
        self,
        api_key: str,
        *,
        api_base: str = DEFAULT_OPENROUTER_API_BASE,
        timeout: float = 120.0,
        app_name: str = "Fyrenzium",
        app_url: str | None = None,
        user_agent: str = "Fyrenzium/1.0",
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.app_name = app_name
        self.app_url = app_url
        self.user_agent = user_agent

    @classmethod
    def from_env(
        cls,
        env_var: str = "OPENROUTER_API_KEY",
        **kwargs: Any,
    ) -> "OpenRouterClient":
        api_key = os.environ.get(env_var, "")
        if not api_key:
            raise OpenRouterError(f"missing environment variable: {env_var}")
        return cls(api_key, **kwargs)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self.api_base}{path}"
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
        }
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        if extra_headers:
            headers.update(extra_headers)

        data = json.dumps(body or {}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=data, method=method.upper(), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                payload = response.read().decode("utf-8")
                return json.loads(payload)
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            message = f"OpenRouter API request failed with status {exc.code}"
            if exc.code == 429:
                raise OpenRouterRateLimitError(exc.code, message, body_text) from exc
            raise OpenRouterAPIError(exc.code, message, body_text) from exc
        except urllib.error.URLError as exc:
            raise OpenRouterError(f"OpenRouter request failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise OpenRouterError("OpenRouter returned invalid JSON") from exc

    def chat_completions(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str = DEFAULT_OPENROUTER_MODEL,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: Sequence[str] | None = None,
        response_format: Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Call OpenRouter's OpenAI-compatible chat completions endpoint."""

        body: dict[str, Any] = {
            "model": model,
            "messages": _normalize_messages(messages),
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if stop is not None:
            body["stop"] = list(stop)
        if response_format is not None:
            body["response_format"] = dict(response_format)
        if extra_body:
            body.update(extra_body)

        return self._request_json(
            "POST",
            "/chat/completions",
            body=body,
            extra_headers=extra_headers,
            timeout=timeout,
        )

    def translate_text(
        self,
        source_text: str,
        *,
        system_prompt: str,
        user_prompt: str | None = None,
        model: str = DEFAULT_OPENROUTER_MODEL,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        response_format: Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Convenience helper for text translation prompts.

        The caller still receives the full raw OpenRouter response so downstream
        code can inspect tokens, refusal messages, or structured outputs.
        """

        messages = [{"role": "system", "content": system_prompt}]
        user_content = user_prompt or source_text
        if user_prompt:
            user_content = f"{user_prompt}\n\nSOURCE:\n{source_text}"
        messages.append({"role": "user", "content": user_content})
        return self.chat_completions(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            extra_body=extra_body,
            timeout=timeout,
        )

    @staticmethod
    def extract_message_content(response: Mapping[str, Any]) -> str:
        choices = response.get("choices") or []
        if not choices:
            raise OpenRouterError("response did not contain any choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, Mapping):
                    text = item.get("text")
                    if text:
                        chunks.append(str(text))
            return "".join(chunks)
        if content is None:
            return ""
        return str(content)
