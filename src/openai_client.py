from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from urllib.parse import urljoin, urlparse
from typing import Any, Callable

import requests
from openai import OpenAI


def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(stripped[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            repaired = extract_label_object(stripped)
            if repaired:
                return repaired
    repaired = extract_label_object(stripped)
    if repaired:
        return repaired
    raise ValueError("Respons model tidak berisi objek JSON valid.")


def extract_label_object(text: str) -> dict[str, Any] | None:
    labels_match = re.search(r'"labels"\s*:\s*\[(.*?)(?:\]|\n\s*\})', text, flags=re.DOTALL)
    if labels_match:
        labels = [item.strip() for item in re.findall(r'"([^"]+)"', labels_match.group(1)) if item.strip()]
        if labels:
            return {"labels": labels}

    label_match = re.search(r'"label"\s*:\s*"([^"]+)"', text)
    if label_match:
        return {"label": label_match.group(1).strip()}
    return None


class OpenAIClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        timeout: float = 60,
        hard_timeout: float | None = None,
    ) -> None:
        self.timeout = timeout
        self.hard_timeout = hard_timeout
        self.provider = provider or os.environ.get("MODEL_API_PROVIDER") or "openai_compatible"
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key and self.base_url and is_local_base_url(self.base_url):
            self.api_key = "ollama"
        if not self.api_key and not self.base_url:
            raise RuntimeError("OPENAI_API_KEY belum tersedia di environment.")
        self.client = (
            OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
            if self.api_key
            else None
        )

    def embed_texts(
        self,
        *,
        model: str,
        texts: list[str],
        task_type: str,
        max_retries: int,
    ) -> list[list[float]]:
        if self.client is None:
            return self._embed_texts_no_auth(model=model, texts=texts, max_retries=max_retries)

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = self.client.embeddings.create(model=model, input=texts)
                vectors = [item.embedding for item in response.data]
                if len(vectors) != len(texts) or any(not vector for vector in vectors):
                    raise RuntimeError("Respons embedding OpenAI tidak sesuai jumlah input.")
                return vectors
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Gagal membuat embedding OpenAI: {last_error}") from last_error

    def generate_json(
        self,
        *,
        model: str,
        system_instruction: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        max_retries: int,
        think: bool | str | None = None,
        stream: bool = False,
        stream_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        if self.provider == "ollama_native":
            return self._generate_json_ollama(
                model=model,
                system_instruction=system_instruction,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                think=think,
            )
        if stream:
            return self._generate_json_stream(
                model=model,
                system_instruction=system_instruction,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                max_retries=max_retries,
                think=think,
                stream_callback=stream_callback,
            )
        if self.client is None:
            return self._generate_json_no_auth(
                model=model,
                system_instruction=system_instruction,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                max_retries=max_retries,
                think=think,
            )

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                    extra_body={"think": think} if think is not None else None,
                )
                text = message_content_to_text(response.choices[0].message.content).strip()
                if not text:
                    reasoning = response.choices[0].message.model_extra.get("reasoning")
                    text = message_content_to_text(reasoning).strip() if reasoning else ""
                if not text:
                    raise RuntimeError("Respons OpenAI kosong.")
                return extract_json_object(text)
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Generasi JSON OpenAI gagal: {last_error}") from last_error

    def _generate_json_stream(
        self,
        *,
        model: str,
        system_instruction: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        max_retries: int,
        think: bool | str | None,
        stream_callback: Callable[[str], None] | None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                if self.client is not None:
                    text = self._stream_openai_sdk(
                        model=model,
                        system_instruction=system_instruction,
                        prompt=prompt,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        think=think,
                        stream_callback=stream_callback,
                    )
                else:
                    text = self._stream_openai_compatible_no_auth(
                        model=model,
                        system_instruction=system_instruction,
                        prompt=prompt,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        think=think,
                        stream_callback=stream_callback,
                    )
                if not text.strip():
                    raise RuntimeError("Respons stream kosong.")
                return extract_json_object(text)
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Generasi JSON stream gagal: {last_error}") from last_error

    def _stream_openai_sdk(
        self,
        *,
        model: str,
        system_instruction: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        think: bool | str | None,
        stream_callback: Callable[[str], None] | None,
    ) -> str:
        chunks: list[str] = []
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
            stream=True,
            extra_body={"think": think} if think is not None else None,
        )
        for event in response:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            text = message_content_to_text(getattr(delta, "content", "") or "")
            if not text:
                reasoning = getattr(delta, "model_extra", {}).get("reasoning") if getattr(delta, "model_extra", None) else ""
                text = message_content_to_text(reasoning) if reasoning else ""
            if text:
                chunks.append(text)
                if stream_callback:
                    stream_callback(text)
        return "".join(chunks)

    def _stream_openai_compatible_no_auth(
        self,
        *,
        model: str,
        system_instruction: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        think: bool | str | None,
        stream_callback: Callable[[str], None] | None,
    ) -> str:
        chunks: list[str] = []
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "stream": True,
        }
        if think is not None:
            payload["think"] = think
        with requests.post(
            openai_compatible_url(self.base_url or "", "/chat/completions"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer ollama"},
            json=payload,
            timeout=(min(10.0, self.timeout), self.timeout),
            stream=True,
        ) as response:
            response.raise_for_status()
            started_at = time.monotonic()
            buffer = ""
            for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
                if self.hard_timeout and time.monotonic() - started_at > self.hard_timeout:
                    raise TimeoutError(
                        f"Request stream OpenAI-compatible melebihi hard timeout "
                        f"{self.hard_timeout:.1f} detik."
                    )
                if not chunk:
                    continue
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if parse_openai_stream_line(line, chunks, stream_callback):
                        return "".join(chunks)
        return "".join(chunks)

    def _embed_texts_no_auth(
        self,
        *,
        model: str,
        texts: list[str],
        max_retries: int,
    ) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                data = post_json(
                    url=openai_compatible_url(self.base_url or "", "/embeddings"),
                    payload={"model": model, "input": texts},
                    timeout=self.timeout,
                )
                response_rows = data.get("data", [])
                vectors = [row.get("embedding") for row in response_rows if isinstance(row, dict)]
                if len(vectors) != len(texts) or any(not isinstance(vector, list) for vector in vectors):
                    raise RuntimeError("Respons embedding OpenAI-compatible tidak sesuai jumlah input.")
                return vectors
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Gagal membuat embedding OpenAI-compatible tanpa API key: {last_error}") from last_error

    def _generate_json_no_auth(
        self,
        *,
        model: str,
        system_instruction: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        max_retries: int,
        think: bool | str | None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                data = post_json(
                    url=openai_compatible_url(self.base_url or "", "/chat/completions"),
                    payload={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": temperature,
                        "max_tokens": max_output_tokens,
                        "stream": False,
                        "think": think,
                    },
                    timeout=self.timeout,
                    headers={"Authorization": "Bearer ollama"},
                )
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError(f"Respons chat tidak berisi choices: {json.dumps(data, ensure_ascii=False)}")
                message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                text = message_content_to_text(message.get("content", "")).strip()
                if not text:
                    reasoning = message.get("reasoning")
                    text = message_content_to_text(reasoning).strip() if reasoning else ""
                if not text:
                    raise RuntimeError("Respons OpenAI-compatible kosong.")
                return extract_json_object(text)
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Generasi JSON OpenAI-compatible tanpa API key gagal: {last_error}") from last_error

    def _generate_json_ollama(
        self,
        *,
        model: str,
        system_instruction: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        think: bool | str | None,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": think if think is not None else False,
            "options": {
                "temperature": temperature,
                "num_predict": max_output_tokens,
            },
        }
        url = ollama_native_url(self.base_url or "http://localhost:11434/v1", "/api/chat")
        data = post_json_with_hard_timeout(
            url=url,
            payload=payload,
            timeout=self.timeout,
            hard_timeout=self.hard_timeout,
        )
        text = str(data.get("message", {}).get("content", "")).strip()
        if not text:
            raise RuntimeError("Respons Ollama kosong.")
        return extract_json_object(text)


def is_local_base_url(base_url: str) -> bool:
    host = urlparse(base_url).hostname
    return host in {"localhost", "127.0.0.1", "::1"}


def ollama_native_url(base_url: str, path: str) -> str:
    parsed = urlparse(base_url)
    scheme = parsed.scheme or "http"
    netloc = parsed.netloc or parsed.path
    return f"{scheme}://{netloc}{path}"


def openai_compatible_url(base_url: str, path: str) -> str:
    if not base_url:
        raise RuntimeError("base_url OpenAI-compatible belum diatur.")
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def parse_openai_stream_line(
    line: str,
    chunks: list[str],
    stream_callback: Callable[[str], None] | None,
) -> bool:
    line = line.strip()
    if not line:
        return False
    if line.startswith("data:"):
        line = line[len("data:") :].strip()
    if line == "[DONE]":
        return True
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return False
    choices = data.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return False
    delta = choices[0].get("delta", {}) or {}
    text = message_content_to_text(delta.get("content", ""))
    if not text:
        text = message_content_to_text(delta.get("reasoning", ""))
    if text:
        chunks.append(text)
        if stream_callback:
            stream_callback(text)
    return False


def post_json_with_hard_timeout(
    *,
    url: str,
    payload: dict[str, Any],
    timeout: float,
    hard_timeout: float | None,
) -> dict[str, Any]:
    effective_timeout = timeout
    if hard_timeout and hard_timeout > 0:
        effective_timeout = min(timeout, hard_timeout)
    return post_json(url=url, payload=payload, timeout=effective_timeout)


def post_json(
    *,
    url: str,
    payload: dict[str, Any],
    timeout: float,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    response = requests.post(url, headers=request_headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Respons JSON tidak berupa objek.")
    return data


def post_json_urllib(
    *,
    url: str,
    payload: dict[str, Any],
    timeout: float,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("Respons JSON tidak berupa objek.")
    return data
