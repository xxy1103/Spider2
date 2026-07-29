"""Logging helpers that keep configured credentials out of diagnostics."""

from __future__ import annotations

import logging
from collections.abc import Iterable


class RedactingFilter(logging.Filter):
    def __init__(self, sensitive_values: Iterable[str | None]) -> None:
        super().__init__()
        self._sensitive_values = tuple(
            value for value in sensitive_values if isinstance(value, str) and value
        )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for value in self._sensitive_values:
            message = message.replace(value, "***REDACTED***")
        record.msg = message
        record.args = ()
        return True


def configured_sensitive_values(config) -> list[str | None]:
    values = [config.secrets["model_api"].get("api_key")]
    snowflake = config.secrets.get("snowflake", {})
    values.extend([snowflake.get("user"), snowflake.get("password")])
    return values
