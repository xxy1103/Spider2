"""Structured, task-scoped tools for Spider Agent TC."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import sqlglot
from sqlglot import exp


PLACEHOLDER_RE = re.compile(
    r"\.\.\.|\bTODO\b|all\s+(?:remaining|daily)\s+tables|remaining\s+tables",
    re.IGNORECASE,
)
DATE_SUFFIX_RE = re.compile(r"(\d{8})$")


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., dict[str, Any]]

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _object_schema(
    properties: dict[str, Any], required: list[str] | None = None
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _json_content(payload: dict[str, Any]) -> dict[str, Any]:
    return {"content": json.dumps(payload, ensure_ascii=False, default=str)}


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_exact_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def _safe_component(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError("Invalid task context identifier")
    return value


def _cursor_offset(cursor: str | None) -> int:
    if cursor is None:
        return 0
    if not re.fullmatch(r"\d+", cursor):
        raise ValueError("cursor must be a non-negative integer string")
    return int(cursor)


def _extract_nested_paths(value: Any, prefix: str = "") -> set[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                return set()
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.add(path)
            paths.update(_extract_nested_paths(child, path))
    elif isinstance(value, list):
        for child in value[:3]:
            paths.update(_extract_nested_paths(child, prefix))
    return paths


def _bounded_sample_row(
    sample: dict[str, Any], max_chars: int
) -> tuple[dict[str, Any], list[str]]:
    serialized = json.dumps(sample, ensure_ascii=False, default=str)
    if len(serialized) <= max_chars:
        return sample, []
    bounded: dict[str, Any] = {}
    truncated_fields: list[str] = []
    for key, value in sample.items():
        value_text = json.dumps(value, ensure_ascii=False, default=str)
        candidate = {**bounded, str(key): value}
        if len(json.dumps(candidate, ensure_ascii=False, default=str)) <= max_chars:
            bounded[str(key)] = value
            continue
        remaining = max(
            0,
            max_chars
            - len(json.dumps(bounded, ensure_ascii=False, default=str))
            - 160,
        )
        bounded[str(key)] = {
            "_truncated": True,
            "original_chars": len(value_text),
            "json_preview": value_text[:remaining],
        }
        truncated_fields.append(str(key))
        truncated_fields.extend(
            str(remaining_key)
            for remaining_key in list(sample)[list(sample).index(key) + 1 :]
        )
        break
    return bounded, truncated_fields


def _load_ddl_map(schema_dir: Path) -> dict[str, tuple[str, str]]:
    ddl_path = schema_dir / "DDL.csv"
    if not ddl_path.is_file():
        return {}
    csv.field_size_limit(20_000_000)
    result: dict[str, tuple[str, str]] = {}
    with ddl_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            table_name = (row.get("table_name") or "").strip()
            if table_name:
                result[table_name.upper()] = (
                    row.get("description") or "",
                    row.get("DDL") or "",
                )
    return result


def build_catalog(config: Any) -> Path:
    """Rebuild the run-scoped catalog for only databases selected in this run."""
    state_dir = config.experiment_dir / "tool-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    target = state_dir / "schema-index.sqlite"
    temporary = state_dir / ".schema-index.sqlite.tmp"
    temporary.unlink(missing_ok=True)

    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE tables (
                database_id TEXT NOT NULL,
                schema_name TEXT NOT NULL,
                table_name TEXT NOT NULL,
                full_table_name TEXT NOT NULL,
                source_json_path TEXT NOT NULL,
                table_description TEXT NOT NULL,
                ddl_text TEXT NOT NULL,
                PRIMARY KEY (database_id, schema_name, table_name)
            );
            CREATE TABLE columns (
                database_id TEXT NOT NULL,
                schema_name TEXT NOT NULL,
                table_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                column_type TEXT NOT NULL,
                description TEXT NOT NULL
            );
            CREATE INDEX columns_scope_idx
                ON columns(database_id, schema_name, table_name);
            CREATE INDEX columns_name_idx ON columns(database_id, column_name);
            """
        )
        selected_databases = sorted(
            {item["db_id"] for item in config.selected_items}
        )
        for database_id in selected_databases:
            database_dir = config.paths["databases"] / database_id
            if not database_dir.is_dir():
                raise RuntimeError(
                    f"Database metadata directory does not exist: {database_id}"
                )
            for schema_dir in sorted(path for path in database_dir.iterdir() if path.is_dir()):
                ddl_map = _load_ddl_map(schema_dir)
                for source_path in sorted(schema_dir.glob("*.json")):
                    try:
                        data = json.loads(source_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        raise RuntimeError(
                            f"Invalid schema metadata: {source_path}: {exc}"
                        ) from exc
                    if not isinstance(data, dict):
                        raise RuntimeError(
                            f"Schema metadata must be an object: {source_path}"
                        )
                    table_name = source_path.stem
                    full_name = f"{database_id}.{schema_dir.name}.{table_name}"
                    table_description, ddl_text = ddl_map.get(
                        table_name.upper(), ("", "")
                    )
                    connection.execute(
                        "INSERT INTO tables VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            database_id,
                            schema_dir.name,
                            table_name,
                            full_name,
                            str(source_path.resolve()),
                            table_description,
                            ddl_text,
                        ),
                    )
                    names = data.get("column_names") or []
                    types = data.get("column_types") or []
                    descriptions = data.get("description") or []
                    if not isinstance(names, list):
                        names = []
                    for index, name in enumerate(names):
                        connection.execute(
                            "INSERT INTO columns VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                database_id,
                                schema_dir.name,
                                table_name,
                                str(name),
                                str(types[index]) if index < len(types) else "",
                                (
                                    str(descriptions[index])
                                    if index < len(descriptions)
                                    else ""
                                ),
                            ),
                        )
        connection.commit()
    finally:
        connection.close()
    target.unlink(missing_ok=True)
    temporary.replace(target)
    return target


class StructuredToolRuntime:
    def __init__(self) -> None:
        self.config: Any | None = None
        self.catalog_path: Path | None = None
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def configure(self, config: Any) -> None:
        self.config = config
        self.catalog_path = config.experiment_dir / "tool-state" / "schema-index.sqlite"
        if not self.catalog_path.is_file():
            raise RuntimeError(
                f"Schema catalog is missing for this run: {self.catalog_path}"
            )

    def _settings(self, section: str) -> dict[str, Any]:
        if self.config is None:
            raise RuntimeError("Structured tools are not configured")
        return self.config.raw["tools"][section]

    def _context(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("Missing injected task context")
        required = {
            "instance_id",
            "rollout_idx",
            "allowed_database",
            "instruction",
            "external_knowledge",
        }
        if not required.issubset(raw):
            raise ValueError("Incomplete injected task context")
        return raw

    def _task_dir(self, context: dict[str, Any]) -> Path:
        if self.config is None:
            raise RuntimeError("Structured tools are not configured")
        instance_id = _safe_component(str(context["instance_id"]))
        rollout_idx = int(context["rollout_idx"])
        path = (
            self.config.experiment_dir
            / "tool-state"
            / "tasks"
            / instance_id
            / f"rollout-{rollout_idx}"
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _lock(self, context: dict[str, Any]) -> threading.Lock:
        key = f"{context['instance_id']}:{context['rollout_idx']}"
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    @contextmanager
    def _connect_catalog(self):
        if self.catalog_path is None:
            raise RuntimeError("Structured tools are not configured")
        uri = self.catalog_path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def _artifact_path(
        self, context: dict[str, Any], prefix: str, artifact_id: str, suffix: str
    ) -> Path:
        if not re.fullmatch(r"[a-z]+_[0-9a-f]{16}", artifact_id):
            raise ValueError(f"Invalid {prefix} id")
        path = self._task_dir(context) / f"{artifact_id}{suffix}"
        if not path.is_file():
            raise ValueError(f"Unknown {prefix} id for this task and rollout")
        return path

    def _write_atomic(self, path: Path, content: str) -> None:
        temporary = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
        temporary.write_bytes(content.encode("utf-8"))
        temporary.replace(path)

    def get_task_context(self, *, _context: Any) -> dict[str, Any]:
        context = self._context(_context)
        allowed = str(context["allowed_database"])
        with self._connect_catalog() as connection:
            rows = connection.execute(
                """
                SELECT schema_name, COUNT(*) AS table_count,
                       MIN(table_name) AS first_table,
                       MAX(table_name) AS last_table
                FROM tables WHERE database_id = ?
                GROUP BY schema_name ORDER BY schema_name
                """,
                (allowed,),
            ).fetchall()
        knowledge = None
        knowledge_name = context.get("external_knowledge")
        if knowledge_name and self.config is not None:
            candidate = (self.config.paths["documents"] / str(knowledge_name)).resolve()
            try:
                candidate.relative_to(self.config.paths["documents"].resolve())
            except ValueError as exc:
                raise ValueError("External knowledge path is outside documents") from exc
            if candidate.is_file():
                knowledge = candidate.read_text(encoding="utf-8")
        return _json_content(
            {
                "instance_id": context["instance_id"],
                "question": context["instruction"],
                "external_knowledge": knowledge,
                "allowed_database": allowed,
                "schemas": [dict(row) for row in rows],
            }
        )

    def search_schema(
        self,
        *,
        query: str,
        schemas: list[str] | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        _context: Any,
    ) -> dict[str, Any]:
        context = self._context(_context)
        settings = self._settings("catalog")
        actual_limit = limit or settings["page_size"]
        if actual_limit < 1 or actual_limit > settings["max_page_size"]:
            raise ValueError("limit is outside the configured catalog page range")
        offset = _cursor_offset(cursor)
        tokens = [token.lower() for token in re.findall(r"[\w.]+", query) if token]
        if not tokens:
            raise ValueError("query must contain searchable text")

        clauses = []
        parameters: list[Any] = [context["allowed_database"]]
        for token in tokens:
            pattern = f"%{token}%"
            clauses.append(
                "(LOWER(t.table_name) LIKE ? OR LOWER(c.column_name) LIKE ? "
                "OR LOWER(c.description) LIKE ? OR "
                "LOWER(t.table_description) LIKE ?)"
            )
            parameters.extend([pattern, pattern, pattern, pattern])
        schema_clause = ""
        if schemas:
            placeholders = ",".join("?" for _ in schemas)
            schema_clause = f" AND UPPER(t.schema_name) IN ({placeholders})"
            parameters.extend(schema.upper() for schema in schemas)
        parameters.extend([actual_limit + 1, offset])
        statement = f"""
            SELECT t.full_table_name, t.schema_name, t.table_name,
                   t.table_description,
                   c.column_name, c.column_type, c.description
            FROM tables t
            LEFT JOIN columns c
              ON c.database_id=t.database_id
             AND c.schema_name=t.schema_name
             AND c.table_name=t.table_name
            WHERE t.database_id = ?
              AND ({' OR '.join(clauses)})
              {schema_clause}
            ORDER BY t.schema_name, t.table_name, c.column_name
            LIMIT ? OFFSET ?
        """
        with self._connect_catalog() as connection:
            rows = connection.execute(statement, parameters).fetchall()
        has_more = len(rows) > actual_limit
        page = rows[:actual_limit]
        return _json_content(
            {
                "matches": [dict(row) for row in page],
                "next_cursor": str(offset + actual_limit) if has_more else None,
            }
        )

    def describe_table(
        self,
        *,
        table: str,
        include_samples: bool = True,
        cursor: str | None = None,
        _context: Any,
    ) -> dict[str, Any]:
        context = self._context(_context)
        parts = table.split(".")
        if len(parts) != 3:
            raise ValueError("table must use database.schema.table")
        if parts[0].upper() != str(context["allowed_database"]).upper():
            raise ValueError("Table is outside the current task database scope")
        offset = _cursor_offset(cursor)
        settings = self._settings("catalog")
        page_size = settings["page_size"]
        with self._connect_catalog() as connection:
            table_row = connection.execute(
                """
                SELECT * FROM tables
                WHERE UPPER(database_id)=UPPER(?)
                  AND UPPER(schema_name)=UPPER(?)
                  AND UPPER(table_name)=UPPER(?)
                """,
                tuple(parts),
            ).fetchone()
            if table_row is None:
                raise ValueError("Table is not present in the current schema catalog")
            columns = connection.execute(
                """
                SELECT column_name, column_type, description FROM columns
                WHERE database_id=? AND schema_name=? AND table_name=?
                ORDER BY rowid LIMIT ? OFFSET ?
                """,
                (
                    table_row["database_id"],
                    table_row["schema_name"],
                    table_row["table_name"],
                    page_size + 1,
                    offset,
                ),
            ).fetchall()
        has_more = len(columns) > page_size
        samples: list[Any] = []
        sample_truncations: list[dict[str, Any]] = []
        nested_paths: dict[str, list[str]] = {}
        if include_samples:
            data = json.loads(
                Path(table_row["source_json_path"]).read_text(encoding="utf-8")
            )
            raw_samples = data.get("sample_rows") or []
            if isinstance(raw_samples, list):
                for sample_index, sample in enumerate(
                    raw_samples[: settings["sample_rows"]]
                ):
                    if isinstance(sample, dict):
                        for key, value in sample.items():
                            paths = sorted(_extract_nested_paths(value))
                            if paths:
                                nested_paths.setdefault(str(key), [])
                                nested_paths[str(key)] = sorted(
                                    set(nested_paths[str(key)]) | set(paths)
                                )
                        bounded, truncated_fields = _bounded_sample_row(
                            sample, settings["max_sample_chars"]
                        )
                        samples.append(bounded)
                        if truncated_fields:
                            sample_truncations.append(
                                {
                                    "sample_index": sample_index,
                                    "fields": truncated_fields,
                                }
                            )
                    else:
                        samples.append(sample)
        return _json_content(
            {
                "table": table_row["full_table_name"],
                "description": table_row["table_description"],
                "ddl": table_row["ddl_text"],
                "columns": [dict(row) for row in columns[:page_size]],
                "next_cursor": str(offset + page_size) if has_more else None,
                "nested_paths": nested_paths,
                "sample_rows": samples,
                "sample_truncations": sample_truncations,
            }
        )

    def resolve_table_set(
        self,
        *,
        schema: str,
        prefix: str,
        start_date: str,
        end_date: str,
        _context: Any,
    ) -> dict[str, Any]:
        context = self._context(_context)
        if not re.fullmatch(r"\d{8}", start_date) or not re.fullmatch(
            r"\d{8}", end_date
        ):
            raise ValueError("start_date and end_date must be YYYYMMDD")
        if start_date > end_date:
            raise ValueError("start_date must not be after end_date")
        with self._connect_catalog() as connection:
            rows = connection.execute(
                """
                SELECT table_name, full_table_name FROM tables
                WHERE database_id=? AND UPPER(schema_name)=UPPER(?)
                  AND SUBSTR(table_name, 1, ?) = ?
                ORDER BY table_name
                """,
                (context["allowed_database"], schema, len(prefix), prefix),
            ).fetchall()
        selected = []
        for row in rows:
            match = DATE_SUFFIX_RE.search(row["table_name"])
            if match and start_date <= match.group(1) <= end_date:
                selected.append(dict(row))
        if not selected:
            raise ValueError("No tables matched the requested scoped date range")
        payload = {
            "instance_id": context["instance_id"],
            "rollout_idx": context["rollout_idx"],
            "allowed_database": context["allowed_database"],
            "schema": schema,
            "prefix": prefix,
            "start_date": start_date,
            "end_date": end_date,
            "tables": [row["full_table_name"] for row in selected],
        }
        digest = _sha256(json.dumps(payload, sort_keys=True))[:16]
        artifact_id = f"tableset_{digest}"
        path = self._task_dir(context) / f"{artifact_id}.json"
        with self._lock(context):
            self._write_atomic(
                path, json.dumps(payload, ensure_ascii=False, indent=2)
            )
        available_dates = {
            DATE_SUFFIX_RE.search(row["table_name"]).group(1) for row in selected
        }
        from datetime import datetime, timedelta

        current = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        missing = []
        while current <= end:
            value = current.strftime("%Y%m%d")
            if value not in available_dates:
                missing.append(value)
            current += timedelta(days=1)
        return _json_content(
            {
                "table_set_id": artifact_id,
                "matched_tables": len(selected),
                "first_table": selected[0]["full_table_name"],
                "last_table": selected[-1]["full_table_name"],
                "missing_dates": missing,
            }
        )

    def _load_table_set(
        self, context: dict[str, Any], artifact_id: str
    ) -> dict[str, Any]:
        path = self._artifact_path(context, "table_set", artifact_id, ".json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = (
            payload.get("instance_id") == context["instance_id"]
            and payload.get("rollout_idx") == context["rollout_idx"]
            and payload.get("allowed_database") == context["allowed_database"]
        )
        if not expected:
            raise ValueError("Table set belongs to another task or rollout")
        return payload

    def build_union_sql(
        self,
        *,
        table_set_id: str,
        select_template: str,
        final_query_template: str | None = None,
        _context: Any,
    ) -> dict[str, Any]:
        context = self._context(_context)
        table_set = self._load_table_set(context, table_set_id)
        if select_template.count("{table}") != 1:
            raise ValueError("select_template must contain {table} exactly once")
        if PLACEHOLDER_RE.search(select_template):
            raise ValueError("select_template contains a forbidden placeholder")
        fragments = [
            select_template.replace("{table}", table)
            for table in table_set["tables"]
        ]
        union_sql = "\nUNION ALL\n".join(fragments)
        if final_query_template is not None:
            if final_query_template.count("{union_sql}") != 1:
                raise ValueError(
                    "final_query_template must contain {union_sql} exactly once"
                )
            sql = final_query_template.replace("{union_sql}", union_sql)
        else:
            sql = union_sql
        validated = self._validate_sql(sql, context)
        digest = validated["sql_sha256"]
        sql_id = f"sql_{digest[:16]}"
        path = self._task_dir(context) / f"{sql_id}.sql"
        with self._lock(context):
            self._write_atomic(path, sql)
        return _json_content(
            {
                "sql_id": sql_id,
                "sql_sha256": digest,
                "sql_chars": len(sql),
                "referenced_tables": len(validated["tables"]),
                "contains_placeholders": False,
                "parse_status": "valid",
            }
        )

    def _validate_sql(
        self, sql: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        settings = self._settings("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("SQL must be a non-empty string")
        if len(sql) > settings["max_sql_chars"]:
            raise ValueError(
                f"SQL exceeds max_sql_chars ({settings['max_sql_chars']})"
            )
        if PLACEHOLDER_RE.search(sql):
            raise ValueError("SQL contains a forbidden placeholder")
        try:
            expressions = sqlglot.parse(sql, read="snowflake")
        except sqlglot.errors.ParseError as exc:
            raise ValueError(f"Snowflake SQL parsing failed: {exc}") from exc
        if len(expressions) != 1 or expressions[0] is None:
            raise ValueError("Exactly one SQL statement is required")
        expression = expressions[0]
        if not isinstance(expression, exp.Query):
            raise ValueError("Only a read-only query statement is allowed")
        prohibited = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter)
        if any(expression.find(node_type) is not None for node_type in prohibited):
            raise ValueError("DDL and DML statements are not allowed")
        cte_names = {
            cte.alias_or_name.upper()
            for cte in expression.find_all(exp.CTE)
            if cte.alias_or_name
        }
        tables = []
        allowed = str(context["allowed_database"]).upper()
        for table in expression.find_all(exp.Table):
            name = table.name
            database = table.catalog
            schema = table.db
            if not name:
                continue
            if not database and not schema and name.upper() in cte_names:
                continue
            if not database or not schema:
                raise ValueError(
                    f"Table must use database.schema.table: {table.sql()}"
                )
            if database.upper() != allowed:
                raise ValueError(
                    f"Table is outside current database scope {allowed}: {table.sql()}"
                )
            if schema.upper() == "INFORMATION_SCHEMA":
                raise ValueError("INFORMATION_SCHEMA queries are not allowed")
            tables.append(f"{database}.{schema}.{name}")
        return {
            "sql_sha256": _sha256(sql),
            "tables": sorted(set(tables)),
        }

    def _load_sql(
        self, context: dict[str, Any], sql: str | None, sql_id: str | None
    ) -> tuple[str, str]:
        if (sql is None) == (sql_id is None):
            raise ValueError("Provide exactly one of sql or sql_id")
        if sql_id is not None:
            path = self._artifact_path(context, "sql", sql_id, ".sql")
            return _read_exact_text(path), sql_id
        assert sql is not None
        digest = _sha256(sql)
        generated_id = f"sql_{digest[:16]}"
        return sql, generated_id

    def _save_execution(
        self, context: dict[str, Any], payload: dict[str, Any]
    ) -> None:
        path = self._task_dir(context) / f"{payload['execution_id']}.json"
        with self._lock(context):
            self._write_atomic(
                path, json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            )

    def _redact_error(self, error: Exception) -> str:
        message = str(error)
        if self.config is None:
            return message
        model_api = self.config.secrets.get("model_api", {})
        snowflake = self.config.secrets.get("snowflake", {})
        for value in (
            model_api.get("api_key"),
            snowflake.get("user"),
            snowflake.get("password"),
        ):
            if isinstance(value, str) and value:
                message = message.replace(value, "***REDACTED***")
        return message

    def _execute_live(
        self, sql: str, preview_rows: int
    ) -> tuple[list[str], list[list[Any]], bool, str | None]:
        import snowflake.connector

        if self.config is None:
            raise RuntimeError("Structured tools are not configured")
        settings = self._settings("sql")
        connection = snowflake.connector.connect(
            **self.config.secrets["snowflake"],
            login_timeout=settings["timeout_seconds"],
            network_timeout=settings["timeout_seconds"],
        )
        try:
            cursor = connection.cursor()
            try:
                cursor.execute(sql, timeout=settings["timeout_seconds"])
                query_id = getattr(cursor, "sfqid", None)
                columns = (
                    [item[0] for item in cursor.description]
                    if cursor.description
                    else []
                )
                rows = cursor.fetchmany(preview_rows + 1) if cursor.description else []
                return columns, [list(row) for row in rows[:preview_rows]], (
                    len(rows) > preview_rows
                ), query_id
            finally:
                cursor.close()
        finally:
            connection.close()

    def _execute_mock(
        self, preview_rows: int
    ) -> tuple[list[str], list[list[Any]], bool, str]:
        mock = self._settings("sql").get("mock", {})
        rows = list(csv.reader((mock.get("response_csv") or "").splitlines()))
        if not rows:
            return [], [], False, "MOCK"
        return rows[0], rows[1 : preview_rows + 1], len(rows) - 1 > preview_rows, "MOCK"

    def execute_sql(
        self,
        *,
        sql: str | None = None,
        sql_id: str | None = None,
        _context: Any,
    ) -> dict[str, Any]:
        context = self._context(_context)
        actual_sql, actual_sql_id = self._load_sql(context, sql, sql_id)
        validated = self._validate_sql(actual_sql, context)
        if sql is not None:
            path = self._task_dir(context) / f"{actual_sql_id}.sql"
            with self._lock(context):
                self._write_atomic(path, actual_sql)
        started = time.monotonic()
        execution_id = f"execution_{uuid.uuid4().hex[:16]}"
        try:
            preview_rows = self._settings("sql")["preview_rows"]
            if self._settings("sql")["mode"] == "mock":
                columns, rows, has_more, query_id = self._execute_mock(preview_rows)
                mock_run = True
            else:
                columns, rows, has_more, query_id = self._execute_live(
                    actual_sql, preview_rows
                )
                mock_run = False
            payload = {
                "execution_id": execution_id,
                "sql_id": actual_sql_id,
                "sql_sha256": validated["sql_sha256"],
                "status": "success",
                "has_rows": bool(rows),
                "columns": columns,
                "preview": [dict(zip(columns, row)) for row in rows],
                "preview_rows": len(rows),
                "more_rows_available": has_more,
                "query_id": query_id,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "mock_run": mock_run,
            }
        except Exception as exc:
            payload = {
                "execution_id": execution_id,
                "sql_id": actual_sql_id,
                "sql_sha256": validated["sql_sha256"],
                "status": "error",
                "has_rows": False,
                "error_type": type(exc).__name__,
                "error": self._redact_error(exc),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "mock_run": self._settings("sql")["mode"] == "mock",
            }
        self._save_execution(context, payload)
        return _json_content(payload)

    def read_query_result(
        self,
        *,
        execution_id: str,
        offset: int,
        limit: int = 20,
        _context: Any,
    ) -> dict[str, Any]:
        context = self._context(_context)
        if offset < 0:
            raise ValueError("offset must be non-negative")
        settings = self._settings("sql")
        if limit < 1 or limit > settings["max_page_size"]:
            raise ValueError("limit is outside the configured result page range")
        execution_path = self._artifact_path(
            context, "execution", execution_id, ".json"
        )
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        if execution.get("status") != "success":
            raise ValueError("Only a successful execution can be paged")
        self._artifact_path(context, "sql", execution["sql_id"], ".sql")
        if settings["mode"] == "mock":
            columns, all_rows, _, _ = self._execute_mock(offset + limit + 1)
            page = all_rows[offset : offset + limit]
            has_more = len(all_rows) > offset + limit
        else:
            query_id = execution.get("query_id")
            if not isinstance(query_id, str) or not re.fullmatch(
                r"[A-Za-z0-9_-]+", query_id
            ):
                raise ValueError("Execution has no valid Snowflake query id")
            paged_sql = (
                f"SELECT * FROM TABLE(RESULT_SCAN('{query_id}')) "
                f"LIMIT {limit + 1} OFFSET {offset}"
            )
            self._validate_sql(paged_sql, context)
            columns, rows, _, _ = self._execute_live(paged_sql, limit + 1)
            page = rows[:limit]
            has_more = len(rows) > limit
        return _json_content(
            {
                "execution_id": execution_id,
                "offset": offset,
                "rows": [dict(zip(columns, row)) for row in page],
                "more_rows_available": has_more,
                "next_offset": offset + limit if has_more else None,
            }
        )

    def get_sql_text(self, *, sql_id: str, _context: Any) -> dict[str, Any]:
        context = self._context(_context)
        path = self._artifact_path(context, "sql", sql_id, ".sql")
        sql = _read_exact_text(path)
        return _json_content(
            {
                "sql_id": sql_id,
                "sql_sha256": _sha256(sql),
                "sql_chars": len(sql),
                "sql": sql,
            }
        )

    def terminate(self, *, answer: str, _context: Any) -> dict[str, Any]:
        context = self._context(_context)
        try:
            validated = self._validate_sql(answer, context)
        except ValueError as exc:
            return _json_content({"accepted": False, "reason": str(exc)})
        settings = self._settings("submission")
        matching_execution = None
        for path in self._task_dir(context).glob("execution_*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("sql_sha256") == validated["sql_sha256"]
                and payload.get("status") == "success"
                and not payload.get("mock_run")
            ):
                sql_path = self._artifact_path(
                    context, "sql", payload["sql_id"], ".sql"
                )
                if _read_exact_text(sql_path) == answer:
                    matching_execution = payload
                    break
        if settings["require_executed"] and matching_execution is None:
            return _json_content(
                {
                    "accepted": False,
                    "reason": (
                        "The submitted SQL does not match a successfully executed "
                        "non-Mock SQL query for this task and rollout."
                    ),
                }
            )
        if (
            settings["reject_empty"]
            and matching_execution is not None
            and not matching_execution.get("has_rows")
        ):
            return _json_content(
                {"accepted": False, "reason": "The matching execution returned no rows."}
            )
        return _json_content(
            {
                "accepted": True,
                "sql_sha256": validated["sql_sha256"],
                "execution_id": (
                    matching_execution.get("execution_id")
                    if matching_execution
                    else None
                ),
            }
        )


RUNTIME = StructuredToolRuntime()


TOOL_DEFINITIONS = [
    ToolDefinition(
        "get_task_context",
        "Get the current question, external knowledge, allowed database, schemas, and table layout summary.",
        _object_schema({}),
        RUNTIME.get_task_context,
    ),
    ToolDefinition(
        "search_schema",
        "Search tables, columns, types, and descriptions only within the current task database.",
        _object_schema(
            {
                "query": {"type": "string"},
                "schemas": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer", "minimum": 1},
                "cursor": {"type": "string"},
            },
            ["query"],
        ),
        RUNTIME.search_schema,
    ),
    ToolDefinition(
        "describe_table",
        "Describe one fully qualified table in the current database, including columns, DDL, nested paths, and bounded samples.",
        _object_schema(
            {
                "table": {"type": "string"},
                "include_samples": {"type": "boolean"},
                "cursor": {"type": "string"},
            },
            ["table"],
        ),
        RUNTIME.describe_table,
    ),
    ToolDefinition(
        "resolve_table_set",
        "Resolve date-suffixed tables in the current database into a task-scoped table set.",
        _object_schema(
            {
                "schema": {"type": "string"},
                "prefix": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
            },
            ["schema", "prefix", "start_date", "end_date"],
        ),
        RUNTIME.resolve_table_set,
    ),
    ToolDefinition(
        "build_union_sql",
        "Build complete UNION ALL SQL from a scoped table set and optionally embed it in a final query template.",
        _object_schema(
            {
                "table_set_id": {"type": "string"},
                "select_template": {"type": "string"},
                "final_query_template": {"type": "string"},
            },
            ["table_set_id", "select_template"],
        ),
        RUNTIME.build_union_sql,
    ),
    ToolDefinition(
        "execute_sql",
        "Validate and execute one read-only Snowflake SQL query or SQL artifact, returning a bounded preview and execution evidence.",
        {
            **_object_schema(
                {"sql": {"type": "string"}, "sql_id": {"type": "string"}}
            ),
            "oneOf": [{"required": ["sql"]}, {"required": ["sql_id"]}],
        },
        RUNTIME.execute_sql,
    ),
    ToolDefinition(
        "read_query_result",
        "Read a bounded page from a previous successful query result.",
        _object_schema(
            {
                "execution_id": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1},
            },
            ["execution_id", "offset"],
        ),
        RUNTIME.read_query_result,
    ),
    ToolDefinition(
        "get_sql_text",
        "Return the complete, untruncated SQL text for a task-scoped SQL artifact.",
        _object_schema({"sql_id": {"type": "string"}}, ["sql_id"]),
        RUNTIME.get_sql_text,
    ),
    ToolDefinition(
        "terminate",
        "Submit the complete final SQL. It is accepted only when it exactly matches a successful, non-empty, non-Mock execution.",
        _object_schema({"answer": {"type": "string"}}, ["answer"]),
        RUNTIME.terminate,
    ),
]


def get_openai_tools() -> list[dict[str, Any]]:
    return [definition.openai_schema() for definition in TOOL_DEFINITIONS]


def configure(config: Any) -> None:
    RUNTIME.configure(config)


def register_tools(registry: Any) -> None:
    for definition in TOOL_DEFINITIONS:
        registry.register_tool(definition.name, definition.handler)
