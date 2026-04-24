"""Minimal, maintainable ElevenLabs client helpers.

The pipeline uses this client for speech-to-text transcription and speech
synthesis. The implementation stays intentionally small and flexible so the
orchestration layer can evolve without hard-coding transport details into the
rest of the codebase.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_API_BASE = "https://api.elevenlabs.io"
DEFAULT_SCRIBE_MODEL_ID = "scribe_v2"
DEFAULT_TTS_MODEL_ID = "eleven_multilingual_v2"


class ElevenLabsError(RuntimeError):
    """Base error for ElevenLabs client failures."""


class ElevenLabsAPIError(ElevenLabsError):
    """Raised when the ElevenLabs API returns a non-success response."""

    def __init__(self, status_code: int, message: str, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class ElevenLabsRateLimitError(ElevenLabsAPIError):
    """Raised when the API rate limits a request."""


@dataclass(frozen=True)
class ElevenLabsResponse:
    """Small wrapper for responses that may be bytes or JSON."""

    status_code: int
    content_type: str
    body: bytes | dict[str, Any] | str


def _guess_mime_type(filename: str) -> str:
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"


def _read_audio_source(source: str | Path | bytes, filename: str | None = None) -> tuple[str, bytes]:
    if isinstance(source, bytes):
        resolved_name = filename or "audio.bin"
        return resolved_name, source

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if not path.is_file():
        raise IsADirectoryError(str(path))
    return filename or path.name, path.read_bytes()


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def _build_multipart_form(
    fields: Mapping[str, Any],
    files: Mapping[str, tuple[str, bytes, str] | Sequence[tuple[str, bytes, str]]],
) -> tuple[bytes, str]:
    boundary = base64.urlsafe_b64encode(os.urandom(18)).decode("ascii").rstrip("=")
    parts = []

    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode("utf-8"))
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        parts.append(_stringify(value).encode("utf-8"))
        parts.append(b"\r\n")

    for name, raw_file_value in files.items():
        if isinstance(raw_file_value, tuple):
            file_items = [raw_file_value]
        else:
            file_items = list(raw_file_value)
        for filename, content, content_type in file_items:
            parts.append(f"--{boundary}\r\n".encode("utf-8"))
            parts.append(
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8")
            )
            parts.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
            parts.append(content)
            parts.append(b"\r\n")

    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


class ElevenLabsClient:
    """Small ElevenLabs API client for transcription and synthesis."""

    def __init__(
        self,
        api_key: str,
        *,
        api_base: str = DEFAULT_API_BASE,
        timeout: float = 120.0,
        default_model_id: str = DEFAULT_TTS_MODEL_ID,
        user_agent: str = "Fyrenzium/1.0",
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.default_model_id = default_model_id
        self.user_agent = user_agent

    @classmethod
    def from_env(
        cls,
        env_var: str = "ELEVENLABS_API_KEY",
        **kwargs: Any,
    ) -> "ElevenLabsClient":
        api_key = os.environ.get(env_var, "")
        if not api_key:
            raise ElevenLabsError(f"missing environment variable: {env_var}")
        return cls(api_key, **kwargs)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        multipart_fields: Mapping[str, Any] | None = None,
        multipart_files: Mapping[str, tuple[str, bytes, str] | Sequence[tuple[str, bytes, str]]] | None = None,
        accept_json: bool = True,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ElevenLabsResponse:
        url = f"{self.api_base}{path}"
        headers: dict[str, str] = {
            "xi-api-key": self.api_key,
            "User-Agent": self.user_agent,
        }
        if extra_headers:
            headers.update(extra_headers)

        data: bytes | None = None
        if json_body is not None and (multipart_fields or multipart_files):
            raise ValueError("json_body cannot be combined with multipart data")
        if json_body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
        elif multipart_files is not None:
            payload = multipart_fields or {}
            data, content_type = _build_multipart_form(payload, multipart_files)
            headers["Content-Type"] = content_type
        elif multipart_fields is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            data = urllib.parse.urlencode({k: _stringify(v) for k, v in multipart_fields.items()}).encode(
                "utf-8"
            )

        request = urllib.request.Request(url, data=data, method=method.upper(), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                content = response.read()
                content_type = response.headers.get("Content-Type", "")
                if accept_json or "application/json" in content_type:
                    try:
                        parsed = json.loads(content.decode("utf-8"))
                    except json.JSONDecodeError:
                        parsed = content.decode("utf-8", errors="replace")
                    return ElevenLabsResponse(response.status, content_type, parsed)
                return ElevenLabsResponse(response.status, content_type, content)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message = f"ElevenLabs API request failed with status {exc.code}"
            if exc.code == 429:
                raise ElevenLabsRateLimitError(exc.code, message, body) from exc
            raise ElevenLabsAPIError(exc.code, message, body) from exc
        except urllib.error.URLError as exc:
            raise ElevenLabsError(f"ElevenLabs request failed: {exc.reason}") from exc

    def list_models(self) -> dict[str, Any]:
        response = self._request("GET", "/v1/models", accept_json=True)
        if isinstance(response.body, dict):
            return response.body
        raise ElevenLabsError("ElevenLabs model list did not return JSON")

    def list_voices(self) -> dict[str, Any]:
        response = self._request("GET", "/v1/voices", accept_json=True)
        if isinstance(response.body, dict):
            return response.body
        raise ElevenLabsError("ElevenLabs voice list did not return JSON")

    def transcribe_audio(
        self,
        audio: str | Path | bytes,
        *,
        filename: str | None = None,
        model_id: str = DEFAULT_SCRIBE_MODEL_ID,
        language_code: str | None = None,
        diarize: bool = True,
        num_speakers: int | None = None,
        keyterms: Sequence[str] | None = None,
        extra_fields: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Submit audio for transcription.

        The ElevenLabs STT API response is returned as decoded JSON so callers can
        preserve the raw structure exactly as received.
        """

        audio_name, audio_bytes = _read_audio_source(audio, filename)
        fields: dict[str, Any] = {
            "model_id": model_id,
            "diarize": diarize,
        }
        if language_code:
            fields["language_code"] = language_code
        if num_speakers is not None:
            fields["num_speakers"] = num_speakers
        if keyterms:
            fields["keyterms"] = list(keyterms)
        if extra_fields:
            fields.update(extra_fields)

        response = self._request(
            "POST",
            "/v1/speech-to-text",
            multipart_fields=fields,
            multipart_files={
                "file": (audio_name, audio_bytes, _guess_mime_type(audio_name)),
            },
            accept_json=True,
            timeout=timeout,
        )
        if isinstance(response.body, dict):
            return response.body
        raise ElevenLabsError("transcription did not return JSON")

    def create_voice_clone(
        self,
        *,
        name: str,
        audio_files: Sequence[str | Path | bytes],
        description: str | None = None,
        labels: Mapping[str, str] | None = None,
        remove_background_noise: bool = False,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if not name.strip():
            raise ValueError("name is required")
        if not audio_files:
            raise ValueError("audio_files is required")

        files: list[tuple[str, bytes, str]] = []
        for index, audio_file in enumerate(audio_files, start=1):
            audio_name, audio_bytes = _read_audio_source(audio_file)
            safe_name = audio_name or f"sample_{index}.wav"
            files.append((safe_name, audio_bytes, _guess_mime_type(safe_name)))

        fields: dict[str, Any] = {
            "name": name,
            "remove_background_noise": remove_background_noise,
        }
        if description:
            fields["description"] = description
        if labels:
            fields["labels"] = dict(labels)

        response = self._request(
            "POST",
            "/v1/voices/add",
            multipart_fields=fields,
            multipart_files={"files[]": files},
            accept_json=True,
            timeout=timeout,
        )
        if isinstance(response.body, dict):
            return response.body
        raise ElevenLabsError("voice cloning did not return JSON")

    def synthesize_speech(
        self,
        text: str,
        voice_id: str,
        *,
        model_id: str | None = None,
        output_format: str | None = "mp3_44100_128",
        voice_settings: Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> bytes:
        """Generate speech audio for the given voice."""

        if not text:
            raise ValueError("text is required")
        if not voice_id:
            raise ValueError("voice_id is required")

        body: dict[str, Any] = {
            "text": text,
            "model_id": model_id or self.default_model_id,
        }
        if output_format:
            body["output_format"] = output_format
        if voice_settings:
            body["voice_settings"] = dict(voice_settings)
        if extra_body:
            body.update(extra_body)

        response = self._request(
            "POST",
            f"/v1/text-to-speech/{urllib.parse.quote(voice_id, safe='')}",
            json_body=body,
            accept_json=False,
            timeout=timeout,
        )
        if isinstance(response.body, bytes):
            return response.body
        if isinstance(response.body, str):
            return response.body.encode("utf-8")
        raise ElevenLabsError("speech synthesis did not return audio bytes")

    def save_speech_to_file(
        self,
        text: str,
        voice_id: str,
        output_path: str | Path,
        *,
        model_id: str | None = None,
        output_format: str | None = "mp3_44100_128",
        voice_settings: Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Path:
        audio = self.synthesize_speech(
            text,
            voice_id,
            model_id=model_id,
            output_format=output_format,
            voice_settings=voice_settings,
            extra_body=extra_body,
            timeout=timeout,
        )
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)
        return path
