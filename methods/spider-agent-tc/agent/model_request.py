"""Pure provider-specific model request adaptation."""

from __future__ import annotations

import json
from typing import Any


class ModelRequestConfigError(ValueError):
    """Raised when provider-specific thinking configuration is invalid."""


_THINKING_LEVELS = {
    "gpt": {"none", "minimal", "low", "medium", "high", "xhigh"},
    "gemini": {"none", "minimal", "low", "medium", "high"},
    "deepseek": {"none", "low", "medium", "high", "xhigh", "max"},
}


def validate_thinking_config(
    model_config: dict[str, Any], *, location: str = "model"
) -> None:
    """Validate the optional provider/thinking_level pair without model-name inference."""
    has_provider = "provider" in model_config
    has_level = "thinking_level" in model_config
    if has_provider != has_level:
        missing = "thinking_level" if has_provider else "provider"
        raise ModelRequestConfigError(
            f"{location}.provider and {location}.thinking_level must be configured together; "
            f"missing {location}.{missing}"
        )
    if not has_provider:
        return

    provider = model_config["provider"]
    level = model_config["thinking_level"]
    if not isinstance(provider, str) or provider not in _THINKING_LEVELS:
        supported = ", ".join(sorted(_THINKING_LEVELS))
        raise ModelRequestConfigError(
            f"{location}.provider must be one of: {supported}"
        )
    if not isinstance(level, str) or level not in _THINKING_LEVELS[provider]:
        supported = ", ".join(sorted(_THINKING_LEVELS[provider]))
        raise ModelRequestConfigError(
            f"{location}.thinking_level {level!r} is not supported for {provider}; "
            f"choose one of: {supported}"
        )


def build_model_request_kwargs(
    model_config: dict[str, Any], *, location: str = "model"
) -> dict[str, Any]:
    """Build OpenAI SDK kwargs for the configured provider thinking level."""
    validate_thinking_config(model_config, location=location)
    if "provider" not in model_config:
        return {}
    provider = model_config["provider"]
    level = model_config["thinking_level"]
    if provider in {"gpt", "gemini"}:
        return {"extra_body": {"reasoning_effort": level}}
    if level == "none":
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    effort = level if level in {"low", "high", "max"} else "high"
    return {
        "extra_body": {
            "thinking": {"type": "enabled"},
            "reasoning_effort": effort,
        }
    }


def deepseek_thinking_enabled(model_config: dict[str, Any]) -> bool:
    """Return whether DeepSeek requires reasoning_content round-tripping."""
    return (
        model_config.get("provider") == "deepseek"
        and model_config.get("thinking_level") not in {None, "none"}
    )


def extract_reasoning_content(message: Any, *, preserve: bool = False) -> str:
    """Read provider-returned reasoning from common Chat Completions fields."""
    candidates = [getattr(message, "reasoning_content", None)]
    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict):
        candidates.extend(
            model_extra.get(field)
            for field in ("reasoning_content", "reasoning", "thinking")
        )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate if preserve else candidate.strip()
        if isinstance(candidate, (dict, list)) and candidate:
            return json.dumps(candidate, ensure_ascii=False, default=str)
    return ""
