import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

TC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TC_ROOT))

from servers.tools import snowflake_tool


def test_mock_snowflake_returns_fixture_without_connecting(monkeypatch):
    config = SimpleNamespace(
        raw={
            "tools": {
                "snowflake": {
                    "mode": "mock",
                    "timeout_seconds": 3,
                    "max_output_chars": 100,
                    "mock": {"response_csv": "VALUE\n1\n"},
                }
            }
        },
        secrets={"model_api": {"base_url": "https://api.test/v1", "api_key": "key"}},
    )
    snowflake_tool.configure(config)
    connector = Mock(side_effect=AssertionError("mock mode must not connect"))
    monkeypatch.setattr(snowflake_tool.snowflake.connector, "connect", connector)
    result = snowflake_tool.execute_snowflake_sql("SELECT 1")
    assert "MOCK MODE" in result["content"]
    assert "VALUE\n1" in result["content"]
    connector.assert_not_called()
