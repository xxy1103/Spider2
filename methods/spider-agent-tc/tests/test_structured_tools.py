import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

TC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TC_ROOT))

from servers.structured_tools import (
    StructuredToolRuntime,
    build_catalog,
    get_openai_tools,
)
from agent.auto_evaluator import extract_sql_answers as auto_extract_sql_answers
from convert_to_submission_format import extract_sql_answers


def write_table(root, database, schema, table, columns=None):
    schema_dir = root / database / schema
    schema_dir.mkdir(parents=True, exist_ok=True)
    columns = columns or [
        ("id", "NUMBER", "identifier"),
        ("amount", "NUMBER", "transaction amount"),
    ]
    payload = {
        "table_name": f"{schema}.{table}",
        "table_fullname": f"{database}.{schema}.{table}",
        "column_names": [column[0] for column in columns],
        "column_types": [column[1] for column in columns],
        "description": [column[2] for column in columns],
        "sample_rows": [
            {
                columns[0][0]: 1,
                columns[1][0]: '{"nested": {"value": 2}}',
            }
        ],
    }
    (schema_dir / f"{table}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    ddl_path = schema_dir / "DDL.csv"
    existing = []
    if ddl_path.exists():
        with ddl_path.open("r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    existing.append(
        {
            "table_name": table,
            "description": f"description for {table}",
            "DDL": f'CREATE TABLE {table} ("id" NUMBER, "amount" NUMBER);',
        }
    )
    with ddl_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["table_name", "description", "DDL"]
        )
        writer.writeheader()
        writer.writerows(existing)


def make_config(tmp_path, mode="live"):
    databases = tmp_path / "databases"
    documents = tmp_path / "documents"
    run_dir = tmp_path / "run"
    documents.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    config = SimpleNamespace(
        experiment_dir=run_dir,
        selected_items=[
            {
                "instance_id": "task",
                "db_id": "DB1",
                "instruction": "question",
            }
        ],
        paths={"databases": databases, "documents": documents},
        raw={
            "tools": {
                "catalog": {
                    "page_size": 20,
                    "max_page_size": 100,
                    "sample_rows": 3,
                    "max_sample_chars": 12000,
                },
                "sql": {
                    "mode": mode,
                    "timeout_seconds": 30,
                    "preview_rows": 20,
                    "max_page_size": 100,
                    "max_sql_chars": 300000,
                    **(
                        {"mock": {"response_csv": "VALUE\n" + "\n".join(map(str, range(30)))}}
                        if mode == "mock"
                        else {}
                    ),
                },
                "submission": {
                    "require_executed": True,
                    "reject_empty": True,
                },
            }
        },
        secrets={
            "snowflake": {
                "user": "user",
                "password": "password",
                "account": "account",
                "role": "role",
                "warehouse": "warehouse",
            }
        },
    )
    return config


def context(database="DB1"):
    return {
        "instance_id": "task",
        "rollout_idx": 0,
        "allowed_database": database,
        "instruction": "question",
        "external_knowledge": None,
    }


def content(result):
    return json.loads(result["content"])


def test_catalog_indexes_only_selected_databases_and_enforces_scope(tmp_path):
    config = make_config(tmp_path)
    write_table(config.paths["databases"], "DB1", "S", "T1")
    write_table(config.paths["databases"], "DB2", "S", "T2")
    build_catalog(config)
    runtime = StructuredToolRuntime()
    runtime.configure(config)

    task = content(runtime.get_task_context(_context=context()))
    assert task["allowed_database"] == "DB1"
    assert task["schemas"][0]["table_count"] == 1

    matches = content(
        runtime.search_schema(query="amount", _context=context())
    )["matches"]
    assert {match["full_table_name"] for match in matches} == {"DB1.S.T1"}

    described = content(runtime.describe_table(table="DB1.S.T1", _context=context()))
    assert described["description"] == "description for T1"
    assert described["columns"][1]["column_name"] == "amount"
    assert "nested.value" in described["nested_paths"]["amount"]

    with pytest.raises(ValueError, match="outside"):
        runtime.describe_table(table="DB2.S.T2", _context=context())


def test_large_samples_are_loaded_on_demand_and_bounded(tmp_path):
    config = make_config(tmp_path)
    write_table(config.paths["databases"], "DB1", "S", "T1")
    source = config.paths["databases"] / "DB1" / "S" / "T1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["sample_rows"] = [
        {
            "id": 1,
            "amount": json.dumps(
                {
                    "nested": {"value": 2},
                    "large_geometry": "x" * 1_000_000,
                }
            ),
        }
    ]
    source.write_text(json.dumps(payload), encoding="utf-8")
    build_catalog(config)
    runtime = StructuredToolRuntime()
    runtime.configure(config)

    without_samples = content(
        runtime.describe_table(
            table="DB1.S.T1",
            include_samples=False,
            _context=context(),
        )
    )
    assert without_samples["sample_rows"] == []

    described = content(
        runtime.describe_table(table="DB1.S.T1", _context=context())
    )
    assert len(json.dumps(described["sample_rows"])) < 13_000
    assert described["sample_truncations"][0]["fields"] == ["amount"]
    assert "nested.value" in described["nested_paths"]["amount"]
    assert "large_geometry" in described["nested_paths"]["amount"]


def test_catalog_reports_corrupted_metadata(tmp_path):
    config = make_config(tmp_path)
    schema_dir = config.paths["databases"] / "DB1" / "S"
    schema_dir.mkdir(parents=True)
    (schema_dir / "BROKEN.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Invalid schema metadata"):
        build_catalog(config)


def test_catalog_supports_concurrent_read_only_searches(tmp_path):
    config = make_config(tmp_path)
    for index in range(20):
        write_table(config.paths["databases"], "DB1", "S", f"T{index:02d}")
    build_catalog(config)
    runtime = StructuredToolRuntime()
    runtime.configure(config)

    def search(_):
        return content(
            runtime.search_schema(
                query="amount",
                limit=100,
                _context=context(),
            )
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(search, range(32)))

    assert all(len(result["matches"]) == 20 for result in results)


def test_sf_bq275_regression_builds_all_366_tables_without_ellipsis(tmp_path):
    config = make_config(tmp_path)
    current = date(2020, 1, 1)
    while current <= date(2020, 12, 31):
        write_table(
            config.paths["databases"],
            "DB1",
            "S",
            f"T_{current:%Y%m%d}",
        )
        current += timedelta(days=1)
    build_catalog(config)
    runtime = StructuredToolRuntime()
    runtime.configure(config)

    table_set = content(
        runtime.resolve_table_set(
            schema="S",
            prefix="T_",
            start_date="20200101",
            end_date="20201231",
            _context=context(),
        )
    )
    assert table_set["matched_tables"] == 366
    assert table_set["missing_dates"] == []

    built = content(
        runtime.build_union_sql(
            table_set_id=table_set["table_set_id"],
            select_template=(
                "SELECT id, '"
                + ("x" * 500)
                + "' AS deterministic_padding FROM {table}"
            ),
            final_query_template=(
                "WITH all_rows AS ({union_sql}) SELECT id FROM all_rows"
            ),
            _context=context(),
        )
    )
    assert built["referenced_tables"] == 366
    sql_payload = content(
        runtime.get_sql_text(sql_id=built["sql_id"], _context=context())
    )
    assert sql_payload["sql"].count("UNION ALL") == 365
    assert sql_payload["sql_chars"] > 180_000
    assert "..." not in sql_payload["sql"]
    assert sql_payload["sql_sha256"] == built["sql_sha256"]
    incomplete = content(
        runtime.terminate(
            answer=(
                "SELECT * FROM DB1.S.T_20200101 "
                "-- ... remaining daily tables"
            ),
            _context=context(),
        )
    )
    assert incomplete["accepted"] is False
    assert "placeholder" in incomplete["reason"]


def test_execute_preview_pagination_and_mock_submission_rejection(tmp_path):
    config = make_config(tmp_path, mode="mock")
    write_table(config.paths["databases"], "DB1", "S", "T1")
    build_catalog(config)
    runtime = StructuredToolRuntime()
    runtime.configure(config)

    executed = content(
        runtime.execute_sql(sql="SELECT id FROM DB1.S.T1", _context=context())
    )
    assert executed["status"] == "success"
    assert executed["preview_rows"] == 20
    assert executed["more_rows_available"] is True

    page = content(
        runtime.read_query_result(
            execution_id=executed["execution_id"],
            offset=20,
            limit=5,
            _context=context(),
        )
    )
    assert len(page["rows"]) == 5

    rejected = content(
        runtime.terminate(
            answer="SELECT id FROM DB1.S.T1", _context=context()
        )
    )
    assert rejected["accepted"] is False
    assert "non-Mock" in rejected["reason"]


def test_live_execution_evidence_and_strict_terminate(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    write_table(config.paths["databases"], "DB1", "S", "T1")
    build_catalog(config)
    runtime = StructuredToolRuntime()
    runtime.configure(config)
    monkeypatch.setattr(
        runtime,
        "_execute_live",
        lambda sql, preview_rows: (["ID"], [[1]], False, "query-id"),
    )
    sql = "SELECT id FROM DB1.S.T1\r\n-- 非ASCII精确文本"

    executed = content(runtime.execute_sql(sql=sql, _context=context()))
    assert executed["has_rows"] is True
    accepted = content(runtime.terminate(answer=sql, _context=context()))
    assert accepted["accepted"] is True
    assert accepted["execution_id"] == executed["execution_id"]

    changed = content(
        runtime.terminate(answer=sql + " ", _context=context())
    )
    assert changed["accepted"] is False
    with pytest.raises(ValueError, match="Unknown sql id"):
        runtime.get_sql_text(
            sql_id=executed["sql_id"],
            _context={**context(), "rollout_idx": 1},
        )


def test_empty_live_result_cannot_be_submitted_or_paged(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    write_table(config.paths["databases"], "DB1", "S", "T1")
    build_catalog(config)
    runtime = StructuredToolRuntime()
    runtime.configure(config)
    monkeypatch.setattr(
        runtime,
        "_execute_live",
        lambda sql, preview_rows: (["ID"], [], False, "query-id"),
    )
    sql = "SELECT id FROM DB1.S.T1"

    executed = content(runtime.execute_sql(sql=sql, _context=context()))
    assert executed["status"] == "success"
    assert executed["has_rows"] is False
    rejected = content(runtime.terminate(answer=sql, _context=context()))
    assert rejected["accepted"] is False
    assert "no rows" in rejected["reason"]


def test_execution_errors_are_credential_redacted(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    write_table(config.paths["databases"], "DB1", "S", "T1")
    build_catalog(config)
    runtime = StructuredToolRuntime()
    runtime.configure(config)

    def fail(sql, preview_rows):
        raise RuntimeError("user password failed")

    monkeypatch.setattr(runtime, "_execute_live", fail)
    executed = content(
        runtime.execute_sql(sql="SELECT id FROM DB1.S.T1", _context=context())
    )

    assert executed["status"] == "error"
    assert executed["error"] == "***REDACTED*** ***REDACTED*** failed"
    with pytest.raises(ValueError, match="successful execution"):
        runtime.read_query_result(
            execution_id=executed["execution_id"],
            offset=0,
            _context=context(),
        )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM DB1.OTHER.T2",
        (
            "WITH x AS (SELECT id FROM DB1.S.T1) "
            "SELECT id FROM x"
        ),
        (
            "WITH RECURSIVE nums(n) AS "
            "(SELECT 1 UNION ALL SELECT n + 1 FROM nums WHERE n < 3) "
            "SELECT n FROM nums"
        ),
        (
            "SELECT value FROM DB1.S.T1, "
            "LATERAL FLATTEN(input => payload)"
        ),
        "SELECT seq4() FROM TABLE(GENERATOR(ROWCOUNT => 10))",
    ],
)
def test_sql_validation_allows_supported_snowflake_queries(tmp_path, sql):
    config = make_config(tmp_path)
    write_table(config.paths["databases"], "DB1", "S", "T1")
    build_catalog(config)
    runtime = StructuredToolRuntime()
    runtime.configure(config)

    validated = runtime._validate_sql(sql, context())

    assert validated["sql_sha256"]


@pytest.mark.parametrize(
    "sql,match",
    [
        ("SELECT id FROM DB2.S.T2", "outside"),
        (
            "WITH T2 AS (SELECT 1) SELECT id FROM DB2.S.T2",
            "outside",
        ),
        ("SELECT id FROM S.T1", "database.schema.table"),
        ("SELECT id FROM DB1.INFORMATION_SCHEMA.TABLES", "INFORMATION_SCHEMA"),
        ("DELETE FROM DB1.S.T1", "read-only"),
        ("SELECT 1; SELECT 2", "Exactly one"),
        ("SELECT id FROM DB1.S.T1 -- ... remaining tables", "placeholder"),
    ],
)
def test_sql_validation_rejects_unsafe_or_incomplete_queries(tmp_path, sql, match):
    config = make_config(tmp_path)
    write_table(config.paths["databases"], "DB1", "S", "T1")
    build_catalog(config)
    runtime = StructuredToolRuntime()
    runtime.configure(config)

    with pytest.raises(ValueError, match=match):
        runtime._validate_sql(sql, context())


def test_tool_schema_is_single_complete_public_surface():
    names = {item["function"]["name"] for item in get_openai_tools()}
    assert names == {
        "get_task_context",
        "search_schema",
        "describe_table",
        "resolve_table_set",
        "build_union_sql",
        "execute_sql",
        "read_query_result",
        "get_sql_text",
        "terminate",
    }


def test_long_sql_survives_submission_extraction_byte_for_byte(tmp_path):
    input_dir = tmp_path / "results"
    output_dir = tmp_path / "submission"
    input_dir.mkdir()
    sql = (
        "WITH all_rows AS (\r\n"
        + "\r\nUNION ALL\r\n".join(
            (
                f"SELECT {index} AS id, '"
                + ("x" * 500)
                + f"' AS padding FROM DB1.S.T_{index:08d}"
            )
            for index in range(366)
        )
        + "\r\n)\r\nSELECT id FROM all_rows"
    )
    record = [
        {
            "instance_id": "task",
            "terminated": True,
            "conversation": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "terminate",
                            "arguments": {"answer": sql},
                        }
                    ],
                }
            ],
        }
    ]
    (input_dir / "task.json").write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8"
    )

    extract_sql_answers(input_dir, output_dir)
    auto_result = auto_extract_sql_answers(input_dir)

    assert (output_dir / "task.sql").read_bytes() == sql.encode("utf-8")
    assert auto_result["processed"] == 1
    assert (
        auto_result["submission_dir"] / "task.sql"
    ).read_bytes() == sql.encode("utf-8")
