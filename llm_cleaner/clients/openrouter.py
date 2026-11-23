"""OpenRouter chat-completions cleaner.

Captures everything the harness needs:
- latency (perf_counter delta)
- tokens (usage.prompt_tokens / completion_tokens)
- cost (best-effort: usage.cost when provided by OpenRouter, otherwise None)
- provider (top-level provider field returned by OpenRouter)
- response_format_used ("json_schema" / "json_object" / "text")
- raw response JSON

JSON parsing strategy:
1. Try direct json.loads
2. Strip ``` fences and retry
3. Find first '{'...'}' span and retry
On failure, return a CleanResult with failure_mode="parse_error" and the
raw text preserved in extras["raw_text"].
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional, Sequence

import requests

from llm_cleaner.clients.base import (
    Cleaner,
    CleanResult,
    coerce_confidence,
    normalize_state,
)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = 90
MAX_RETRIES = 3
RATE_LIMIT_BACKOFF = 5  # seconds, doubled on each retry


class OpenRouterCleaner(Cleaner):
    """One instance per model id. The harness creates 6 of these."""

    def __init__(
        self,
        model_id: str,
        *,
        api_key: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        extra_headers: Optional[dict[str, str]] = None,
        session: Optional[requests.Session] = None,
    ):
        self.name = model_id
        self.model_id = model_id
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to .env or the environment."
            )
        self.timeout = timeout
        self.extra_headers = extra_headers or {}
        self.session = session or requests.Session()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/llm-data-cleaning",
            "X-Title": "LLM Data Cleaning Thesis",
        }
        headers.update(self.extra_headers)
        return headers

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        backoff = RATE_LIMIT_BACKOFF
        last_err: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            t0 = time.perf_counter()
            try:
                resp = self.session.post(
                    OPENROUTER_URL,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as e:
                last_err = e
                time.sleep(backoff)
                backoff *= 2
                continue

            latency_ms = int((time.perf_counter() - t0) * 1000)

            if resp.status_code == 429:
                time.sleep(backoff)
                backoff *= 2
                continue
            if 500 <= resp.status_code < 600:
                last_err = requests.HTTPError(f"{resp.status_code}: {resp.text[:300]}")
                time.sleep(backoff)
                backoff *= 2
                continue

            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                raise requests.HTTPError(
                    f"OpenRouter HTTP {resp.status_code}: {resp.text[:500]}"
                ) from e

            try:
                return resp.json(), latency_ms
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"OpenRouter returned non-JSON body: {resp.text[:500]}"
                ) from e

        raise RuntimeError(
            f"OpenRouter call failed after {MAX_RETRIES} attempts: {last_err}"
        )

    def _build_payload(
        self,
        prompt_messages: list[dict[str, Any]],
        *,
        temperature: float,
        seed: Optional[int],
        response_format: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": prompt_messages,
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    def _structured_attempt(
        self,
        prompt_messages: list[dict[str, Any]],
        *,
        temperature: float,
        seed: Optional[int],
        response_format: Optional[dict[str, Any]],
    ) -> tuple[dict[str, Any], int, str]:
        """Call once with json_schema, fall back to json_object, then text."""
        if response_format is not None and response_format.get("type") == "json_schema":
            try:
                data, latency = self._post(
                    self._build_payload(
                        prompt_messages,
                        temperature=temperature,
                        seed=seed,
                        response_format=response_format,
                    )
                )
                return data, latency, "json_schema"
            except requests.HTTPError as e:
                if "structured" not in str(e).lower() and "response_format" not in str(e).lower():
                    raise
                # fall through to json_object

        if response_format is not None:
            try:
                data, latency = self._post(
                    self._build_payload(
                        prompt_messages,
                        temperature=temperature,
                        seed=seed,
                        response_format={"type": "json_object"},
                    )
                )
                return data, latency, "json_object"
            except requests.HTTPError as e:
                if "response_format" not in str(e).lower():
                    raise

        data, latency = self._post(
            self._build_payload(
                prompt_messages,
                temperature=temperature,
                seed=seed,
                response_format=None,
            )
        )
        return data, latency, "text"

    @staticmethod
    def _extract_content(api_response: dict[str, Any]) -> str:
        try:
            return api_response["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return ""

    @staticmethod
    def _strip_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", t)
            t = re.sub(r"\n?```\s*$", "", t)
        return t.strip()

    @staticmethod
    def _try_json(text: str) -> Optional[Any]:
        for candidate in (text, OpenRouterCleaner._strip_fences(text)):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def _usage(api_response: dict[str, Any]) -> tuple[Optional[int], Optional[int], Optional[float]]:
        usage = api_response.get("usage") or {}
        tokens_in = usage.get("prompt_tokens")
        tokens_out = usage.get("completion_tokens")
        cost = usage.get("cost")
        if cost is None:
            cost = usage.get("total_cost")
        try:
            cost_f = float(cost) if cost is not None else None
        except (TypeError, ValueError):
            cost_f = None
        return tokens_in, tokens_out, cost_f

    @staticmethod
    def _provider(api_response: dict[str, Any]) -> Optional[str]:
        prov = api_response.get("provider")
        if isinstance(prov, dict):
            return prov.get("name") or prov.get("id")
        if isinstance(prov, str):
            return prov
        return None

    def _result_from_object(
        self,
        obj: Any,
        api_response: dict[str, Any],
        *,
        latency_ms: int,
        response_format_used: str,
        seed: Optional[int],
    ) -> CleanResult:
        tokens_in, tokens_out, cost = self._usage(api_response)
        provider = self._provider(api_response)
        if not isinstance(obj, dict):
            return CleanResult(
                state=None,
                value=None,
                message=None,
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
                provider=provider,
                response_format_used=response_format_used,
                seed=seed,
                raw_response=api_response,
                failure_mode="parse_error",
                parse_error=f"expected JSON object, got {type(obj).__name__}",
            )

        state = normalize_state(obj.get("state"))
        value = obj.get("value")
        if value is not None and not isinstance(value, str):
            value = str(value)
        message = obj.get("message")
        if message is not None and not isinstance(message, str):
            message = str(message)
        confidence = coerce_confidence(obj.get("confidence"))

        failure_mode = None
        if state is None:
            failure_mode = "invalid_state"

        return CleanResult(
            state=state,
            value=value,
            message=message,
            confidence=confidence,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            provider=provider,
            response_format_used=response_format_used,
            seed=seed,
            raw_response=api_response,
            failure_mode=failure_mode,
        )

    def clean(
        self,
        value: str,
        scope: Sequence[str],
        *,
        prompt_variant: str,
        prompt_messages: Optional[list[dict[str, Any]]] = None,
        response_format: Optional[dict[str, Any]] = None,
        temperature: float = 0.1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> CleanResult:
        if prompt_messages is None:
            raise ValueError(
                "OpenRouterCleaner.clean requires prompt_messages from prompts.py"
            )

        api_response, latency_ms, response_format_used = self._structured_attempt(
            prompt_messages,
            temperature=temperature,
            seed=seed,
            response_format=response_format,
        )
        content = self._extract_content(api_response)
        obj = self._try_json(content) if content else None

        if obj is None:
            tokens_in, tokens_out, cost = self._usage(api_response)
            return CleanResult(
                state=None,
                value=None,
                message=None,
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
                provider=self._provider(api_response),
                response_format_used=response_format_used,
                seed=seed,
                raw_response=api_response,
                failure_mode="parse_error",
                parse_error=f"could not parse JSON from content: {content[:200]!r}",
                extras={"raw_text": content},
            )

        return self._result_from_object(
            obj,
            api_response,
            latency_ms=latency_ms,
            response_format_used=response_format_used,
            seed=seed,
        )

    def clean_batch(
        self,
        values: Sequence[str],
        scope: Sequence[str],
        *,
        prompt_variant: str,
        prompt_messages: Optional[list[dict[str, Any]]] = None,
        response_format: Optional[dict[str, Any]] = None,
        temperature: float = 0.1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> list[CleanResult]:
        if prompt_messages is None:
            raise ValueError(
                "OpenRouterCleaner.clean_batch requires prompt_messages from prompts.py"
            )

        api_response, latency_ms, response_format_used = self._structured_attempt(
            prompt_messages,
            temperature=temperature,
            seed=seed,
            response_format=response_format,
        )
        content = self._extract_content(api_response)
        obj = self._try_json(content) if content else None
        tokens_in, tokens_out, cost = self._usage(api_response)
        provider = self._provider(api_response)

        if isinstance(obj, dict) and "results" in obj and isinstance(obj["results"], list):
            obj = obj["results"]

        if not isinstance(obj, list):
            err = CleanResult(
                state=None,
                value=None,
                message=None,
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
                provider=provider,
                response_format_used=response_format_used,
                seed=seed,
                raw_response=api_response,
                failure_mode="parse_error",
                parse_error=f"batch response was not a JSON array: {content[:200]!r}",
                extras={"raw_text": content},
            )
            return [err for _ in values]

        results: list[CleanResult] = []
        per_call_latency = latency_ms // max(len(values), 1)
        per_call_tokens_in = (tokens_in // max(len(values), 1)) if tokens_in else None
        per_call_tokens_out = (tokens_out // max(len(values), 1)) if tokens_out else None
        per_call_cost = (cost / max(len(values), 1)) if cost else None

        for idx, val in enumerate(values):
            if idx < len(obj):
                results.append(
                    self._result_from_object(
                        obj[idx],
                        api_response,
                        latency_ms=per_call_latency,
                        response_format_used=response_format_used,
                        seed=seed,
                    )
                )
            else:
                results.append(
                    CleanResult(
                        state=None,
                        value=None,
                        message=None,
                        latency_ms=per_call_latency,
                        tokens_in=per_call_tokens_in,
                        tokens_out=per_call_tokens_out,
                        cost_usd=per_call_cost,
                        provider=provider,
                        response_format_used=response_format_used,
                        seed=seed,
                        raw_response=api_response,
                        failure_mode="parse_error",
                        parse_error=f"batch response missing element {idx} of {len(values)}",
                    )
                )
            results[-1].tokens_in = per_call_tokens_in
            results[-1].tokens_out = per_call_tokens_out
            results[-1].cost_usd = per_call_cost
            results[-1].extras = {"batch_index": idx, "batch_size": len(values)}
        return results
