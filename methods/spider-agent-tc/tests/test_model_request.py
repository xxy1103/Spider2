import pytest

from agent.model_request import (
    ModelRequestConfigError,
    build_model_request_kwargs,
)


@pytest.mark.parametrize(
    "provider,level,expected",
    [
        ("gpt", "none", {"extra_body": {"reasoning_effort": "none"}}),
        ("gpt", "xhigh", {"extra_body": {"reasoning_effort": "xhigh"}}),
        ("gemini", "minimal", {"extra_body": {"reasoning_effort": "minimal"}}),
        ("gemini", "high", {"extra_body": {"reasoning_effort": "high"}}),
        ("deepseek", "none", {"extra_body": {"thinking": {"type": "disabled"}}}),
        (
            "deepseek",
            "low",
            {"extra_body": {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}},
        ),
        (
            "deepseek",
            "max",
            {"extra_body": {"thinking": {"type": "enabled"}, "reasoning_effort": "max"}},
        ),
    ],
)
def test_build_model_request_kwargs(provider, level, expected):
    assert build_model_request_kwargs(
        {"provider": provider, "thinking_level": level}
    ) == expected


def test_unconfigured_model_preserves_legacy_request():
    assert build_model_request_kwargs({"name": "custom-model"}) == {}


@pytest.mark.parametrize(
    "config,match",
    [
        ({"provider": "gpt"}, "configured together"),
        ({"thinking_level": "high"}, "configured together"),
        ({"provider": "unknown", "thinking_level": "high"}, "provider must be"),
        ({"provider": "gemini", "thinking_level": "xhigh"}, "not supported"),
        ({"provider": "deepseek", "thinking_level": "minimal"}, "not supported"),
    ],
)
def test_invalid_thinking_configuration_is_rejected(config, match):
    with pytest.raises(ModelRequestConfigError, match=match):
        build_model_request_kwargs(config)
