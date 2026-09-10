"""Small, deterministic guardrails for untrusted prompts and model output."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


PROMPT_INJECTION_MESSAGE = (
    "I couldn't process that request because it contains instructions that may "
    "attempt to override the assistant's safety rules."
)
RAG_QUARANTINE_MESSAGE = (
    "I couldn't use the retrieved course material because it contains "
    "instruction-like content that may be a prompt injection."
)
UNSAFE_OUTPUT_MESSAGE = "I couldn't provide that response because it did not pass the safety check."

MAX_PROMPT_CHARS = 16_000
MAX_OUTPUT_CHARS = 20_000

_INJECTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\b(?:ignore|disregard|forget|override|bypass)\b.{0,80}\b(?:previous|prior|above|system|developer|original)\b.{0,40}\b(?:instructions?|prompts?|messages?|rules?)\b",
        r"\b(?:reveal|show|print|repeat|return|expose)\b.{0,80}\b(?:system|developer|hidden|initial)\b.{0,30}\b(?:prompts?|messages?|instructions?|secrets?)\b",
        r"\b(?:follow|obey)\b.{0,40}\b(?:these|my|the following|new)\b.{0,20}\binstructions?\b.{0,20}\binstead\b",
        r"\b(?:you are now|act as)\b.{0,30}\b(?:system|developer|unrestricted|jailbroken)\b",
        r"(?:<\s*/?\s*(?:system|assistant|developer)\s*>|\[\s*INST\s*\]|###\s*(?:system|developer|assistant))",
        r"\b(?:do not|don't)\b.{0,40}\b(?:answer|follow)\b.{0,40}\b(?:the user|the question|system instructions?)\b",
    )
)

_UNSAFE_OUTPUT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"(?:^|\n)\s*(?:system|developer)\s+(?:prompt|message)\s*:",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"\b(?:OPENAI|OPENROUTER|NAHAJ)_API_KEY\s*[:=]\s*\S+",
        r"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}\b",
        r"<\s*script\b|javascript\s*:|\bon\w+\s*=",
    )
)


class PromptInjectionBlocked(ValueError):
    """Raised when untrusted input looks like an instruction-hijacking attempt."""


class OutputValidationError(ValueError):
    """Raised when assistant output is unsafe to return to the client."""


def _normalized(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    return value.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")


def has_prompt_injection(text: Any) -> bool:
    """Detect common instruction override and prompt-exfiltration patterns."""

    value = _normalized(text)
    return len(value) > MAX_PROMPT_CHARS or any(pattern.search(value) for pattern in _INJECTION_PATTERNS)


def ensure_safe_prompt(text: Any) -> str:
    value = _normalized(text).strip()
    if has_prompt_injection(value):
        raise PromptInjectionBlocked(PROMPT_INJECTION_MESSAGE)
    return value


def filter_retrieved_context(
    context: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Quarantine retrieved chunks that contain likely embedded instructions."""

    safe: list[dict[str, Any]] = []
    blocked = 0
    for chunk in context:
        item = dict(chunk)
        if has_prompt_injection(item.get("text", "")):
            blocked += 1
            continue
        safe.append(item)
    return safe, blocked


def validate_final_response(text: Any) -> str:
    """Normalize and validate the complete assistant response before release."""

    value = _normalized(text)
    value = "".join(char for char in value if char in "\n\t" or ord(char) >= 32).strip()
    if not value:
        raise OutputValidationError("The assistant response is empty.")
    if len(value) > MAX_OUTPUT_CHARS:
        raise OutputValidationError("The assistant response exceeds the configured limit.")
    if any(pattern.search(value) for pattern in _UNSAFE_OUTPUT_PATTERNS):
        raise OutputValidationError("The assistant response contains restricted content.")
    return value


def safe_final_response(text: Any) -> str:
    """Return a stable safe response when final-output validation fails."""

    try:
        return validate_final_response(text)
    except OutputValidationError:
        return UNSAFE_OUTPUT_MESSAGE
