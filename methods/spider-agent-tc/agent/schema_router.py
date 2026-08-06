"""Standalone metadata-only Schema Router and logical table-family catalog."""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from openai import APIConnectionError, APITimeoutError, OpenAI

from .model_request import (
    build_model_request_kwargs,
    deepseek_thinking_enabled,
    extract_reasoning_content,
)


logger = logging.getLogger(__name__)

DATE_SUFFIX_RE = re.compile(r"_(?P<value>(?:17|18|19|20)\d{6})$")
DATE_YM_SUFFIX_RE = re.compile(
    r"_(?P<value>(?:17|18|19|20)\d{2}_?(?:0[1-9]|1[0-2]))$"
)
QUARTER_SUFFIX_RE = re.compile(
    r"_(?P<year>(?:17|18|19|20)\d{2})_?(?P<quarter>Q[1-4])$", re.IGNORECASE
)
PERIOD_SUFFIX_RE = re.compile(
    r"_(?P<year>(?:17|18|19|20)\d{2})_(?P<period>1YR|3YR|5YR)$"
)
YEAR_SUFFIX_RE = re.compile(r"(?P<separator>_?)(?P<value>(?:17|18|19|20)\d{2})$")
UNDERSCORE_VERSION_RE = re.compile(r"_(?P<value>[VR]?\d{1,3})$", re.IGNORECASE)
COMPACT_VERSION_RE = re.compile(r"(?<=[A-Za-z_])(?P<value>\d{2})$")
REL_PREFIX_RE = re.compile(r"^REL(?P<value>\d+)_(?P<rest>.+)$", re.IGNORECASE)
CHR_SUFFIX_RE = re.compile(r"_(?P<value>CHR(?:[1-9]|1\d|2[0-2]|X|Y))$", re.IGNORECASE)
TIERS = {"required", "supporting", "possible"}
ROLES = {"fact", "filter", "dimension", "bridge", "geography", "reference"}
SELECTOR_MODES = {"all", "exact", "date_ranges", "versions"}


class SchemaRouterError(RuntimeError):
    """Base class for Schema Router failures."""


class SelectionValidationError(ValueError):
    """Raised when a model submission is outside the catalog contract."""


@dataclass(frozen=True)
class TableMetadata:
    database_id: str
    schema_name: str
    table_name: str
    full_name: str
    path: Path
    columns: tuple[str, ...]
    column_types: tuple[str, ...]
    column_descriptions: tuple[str, ...]
    table_description: str
    ddl: str
    sample_rows: tuple[Any, ...]


@dataclass(frozen=True)
class TableVariant:
    table: TableMetadata
    value: str


@dataclass(frozen=True)
class TableFamily:
    family_id: str
    database_id: str
    schema_name: str
    logical_name: str
    variant_kind: str
    variants: tuple[TableVariant, ...]
    common_columns: tuple[str, ...]
    differing_columns: tuple[str, ...]
    representative_similarity_min: float
    representative_similarity_median: float

    @property
    def physical_tables(self) -> tuple[str, ...]:
        return tuple(variant.table.full_name for variant in self.variants)


def _load_ddl_map(schema_dir: Path) -> dict[str, tuple[str, str]]:
    import csv

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


def _load_table(
    database_id: str,
    schema_dir: Path,
    path: Path,
    ddl_map: dict[str, tuple[str, str]],
) -> TableMetadata:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaRouterError(f"Invalid schema metadata {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SchemaRouterError(f"Schema metadata must be an object: {path}")
    table_name = path.stem
    table_description, ddl = ddl_map.get(table_name.upper(), ("", ""))
    columns = data.get("column_names") or []
    column_types = data.get("column_types") or []
    descriptions = data.get("description") or []
    samples = data.get("sample_rows") or []
    if not isinstance(columns, list):
        columns = []
    if not isinstance(column_types, list):
        column_types = []
    if not isinstance(descriptions, list):
        descriptions = []
    if not isinstance(samples, list):
        samples = []
    return TableMetadata(
        database_id=database_id,
        schema_name=schema_dir.name,
        table_name=table_name,
        full_name=f"{database_id}.{schema_dir.name}.{table_name}",
        path=path.resolve(),
        columns=tuple(str(value) for value in columns),
        column_types=tuple(str(value) for value in column_types),
        column_descriptions=tuple(str(value) for value in descriptions),
        table_description=table_description,
        ddl=ddl,
        sample_rows=tuple(samples),
    )


def _provisional_family(table_name: str) -> tuple[str, str, str]:
    match = REL_PREFIX_RE.search(table_name)
    if match:
        return (
            f"REL{{VERSION}}_{match.group('rest')}",
            "version_candidate",
            match.group("value"),
        )
    match = CHR_SUFFIX_RE.search(table_name)
    if match:
        return (
            CHR_SUFFIX_RE.sub("_{VERSION}", table_name),
            "version_candidate",
            match.group("value").upper(),
        )
    match = DATE_SUFFIX_RE.search(table_name)
    if match:
        return (
            DATE_SUFFIX_RE.sub("_{YYYYMMDD}", table_name),
            "date",
            match.group("value"),
        )
    match = DATE_YM_SUFFIX_RE.search(table_name)
    if match:
        return (
            DATE_YM_SUFFIX_RE.sub("_{YYYYMM}", table_name),
            "date",
            match.group("value").replace("_", ""),
        )
    match = QUARTER_SUFFIX_RE.search(table_name)
    if match:
        return (
            QUARTER_SUFFIX_RE.sub("_{YEAR}_{QUARTER}", table_name),
            "quarter",
            f"{match.group('year')}_{match.group('quarter').upper()}",
        )
    match = PERIOD_SUFFIX_RE.search(table_name)
    if match:
        return (
            PERIOD_SUFFIX_RE.sub("_{YEAR}_{PERIOD}", table_name),
            "period",
            f"{match.group('year')}_{match.group('period')}",
        )
    match = YEAR_SUFFIX_RE.search(table_name)
    if match:
        separator = match.group("separator")
        return (
            YEAR_SUFFIX_RE.sub(f"{separator}{{YEAR}}", table_name),
            "year",
            match.group("value"),
        )
    match = UNDERSCORE_VERSION_RE.search(table_name)
    if match:
        return (
            UNDERSCORE_VERSION_RE.sub("_{VERSION}", table_name),
            "version_candidate",
            match.group("value"),
        )
    match = COMPACT_VERSION_RE.search(table_name)
    if match:
        return (
            COMPACT_VERSION_RE.sub("{VERSION}", table_name),
            "version_candidate",
            match.group("value"),
        )
    return table_name, "singleton", "current"


def _version_base(logical_name: str) -> str:
    return logical_name.replace("_{VERSION}", "").replace("{VERSION}", "")


def _column_summary(
    tables: list[TableMetadata],
) -> tuple[tuple[str, ...], tuple[str, ...], float, float]:
    column_sets = [
        {column.lower(): column for column in table.columns} for table in tables
    ]
    if not column_sets:
        return (), (), 1.0, 1.0
    common_keys = set(column_sets[0])
    union_keys = set(column_sets[0])
    for values in column_sets[1:]:
        common_keys &= set(values)
        union_keys |= set(values)
    canonical: dict[str, str] = {}
    for values in column_sets:
        canonical.update(values)
    representative = set(column_sets[0])
    similarities = []
    for values in column_sets[1:]:
        current = set(values)
        denominator = len(representative | current)
        similarities.append(
            len(representative & current) / denominator if denominator else 1.0
        )
    return (
        tuple(sorted(canonical[key] for key in common_keys)),
        tuple(sorted(canonical[key] for key in union_keys - common_keys)),
        min(similarities, default=1.0),
        statistics.median(similarities) if similarities else 1.0,
    )


class SchemaRouterCatalog:
    """In-memory canonical catalog used by both the Router and evaluator."""

    def __init__(
        self,
        databases_path: Path,
        database_ids: set[str] | None = None,
    ) -> None:
        self.databases_path = databases_path
        self.tables: dict[str, TableMetadata] = {}
        self.tables_upper: dict[str, str] = {}
        self.families: dict[str, TableFamily] = {}
        self.families_by_database: dict[str, list[str]] = defaultdict(list)
        self.table_to_family: dict[str, str] = {}
        selected = (
            sorted(database_ids)
            if database_ids is not None
            else sorted(path.name for path in databases_path.iterdir() if path.is_dir())
        )
        for database_id in selected:
            database_dir = databases_path / database_id
            if not database_dir.is_dir():
                raise SchemaRouterError(
                    f"Database metadata directory does not exist: {database_id}"
                )
            for schema_dir in sorted(path for path in database_dir.iterdir() if path.is_dir()):
                ddl_map = _load_ddl_map(schema_dir)
                for path in sorted(schema_dir.glob("*.json")):
                    table = _load_table(database_id, schema_dir, path, ddl_map)
                    if table.full_name.upper() in self.tables_upper:
                        raise SchemaRouterError(
                            f"Duplicate case-insensitive table name: {table.full_name}"
                        )
                    self.tables[table.full_name] = table
                    self.tables_upper[table.full_name.upper()] = table.full_name
        self._build_families()

    def _build_families(self) -> None:
        provisional: dict[
            tuple[str, str, str, str], list[tuple[TableMetadata, str]]
        ] = defaultdict(list)
        exact_by_scope: dict[tuple[str, str, str], TableMetadata] = {}
        for table in self.tables.values():
            logical_name, kind, value = _provisional_family(table.table_name)
            provisional[
                (table.database_id, table.schema_name, logical_name, kind)
            ].append((table, value))
            exact_by_scope[
                (table.database_id, table.schema_name, table.table_name)
            ] = table

        consumed: set[str] = set()
        groups: list[
            tuple[str, str, str, str, list[tuple[TableMetadata, str]]]
        ] = []
        for (database_id, schema_name, logical_name, kind), values in sorted(
            provisional.items()
        ):
            if kind != "version_candidate":
                continue
            base = _version_base(logical_name)
            if len(values) < 2:
                continue
            current = exact_by_scope.get((database_id, schema_name, base))
            members = list(values)
            if current is not None:
                members.append((current, "current"))
            groups.append(
                (database_id, schema_name, logical_name, "version", members)
            )
            consumed.update(table.full_name for table, _ in members)

        for (database_id, schema_name, logical_name, kind), values in sorted(
            provisional.items()
        ):
            remaining = [
                (table, value)
                for table, value in values
                if table.full_name not in consumed
            ]
            if not remaining:
                continue
            if kind == "version_candidate":
                for table, _ in remaining:
                    groups.append(
                        (
                            database_id,
                            schema_name,
                            table.table_name,
                            "singleton",
                            [(table, "current")],
                        )
                    )
                continue
            groups.append(
                (database_id, schema_name, logical_name, kind, remaining)
            )

        for database_id, schema_name, logical_name, kind, values in groups:
            values.sort(key=lambda item: item[0].table_name)
            tables = [table for table, _ in values]
            common, differing, minimum, median = _column_summary(tables)
            family_id = f"{database_id}.{schema_name}.{logical_name}"
            if family_id in self.families:
                raise SchemaRouterError(f"Duplicate family id: {family_id}")
            family = TableFamily(
                family_id=family_id,
                database_id=database_id,
                schema_name=schema_name,
                logical_name=logical_name,
                variant_kind=kind,
                variants=tuple(
                    TableVariant(table=table, value=value)
                    for table, value in values
                ),
                common_columns=common,
                differing_columns=differing,
                representative_similarity_min=round(minimum, 6),
                representative_similarity_median=round(median, 6),
            )
            self.families[family_id] = family
            self.families_by_database[database_id].append(family_id)
            for table in tables:
                self.table_to_family[table.full_name] = family_id
        for values in self.families_by_database.values():
            values.sort()

    def canonical_table(self, value: str) -> str | None:
        return self.tables_upper.get(value.upper())

    def database_families(self, database_id: str) -> list[TableFamily]:
        return [
            self.families[family_id]
            for family_id in self.families_by_database.get(database_id, [])
        ]

    def family_overview(self, database_id: str) -> str:
        by_schema: dict[str, list[TableFamily]] = defaultdict(list)
        for family in self.database_families(database_id):
            by_schema[family.schema_name].append(family)
        lines = []
        for schema_name in sorted(by_schema):
            families = by_schema[schema_name]
            physical_count = sum(len(family.variants) for family in families)
            lines.append(
                f"- {schema_name}: {len(families)} families, "
                f"{physical_count} physical tables"
            )
            for family in families:
                values = [variant.value for variant in family.variants]
                preview = ", ".join(values[:5])
                if len(values) > 5:
                    preview += f", ... ({len(values)} total)"
                lines.append(
                    f"  - {family.family_id} "
                    f"[{family.variant_kind}; variants: {preview}]"
                )
        return "\n".join(lines)


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _bounded(value: str, maximum: int) -> tuple[str, bool]:
    if len(value) <= maximum:
        return value, False
    return value[:maximum], True


class SchemaRouterTools:
    """Metadata-only tools exposed to the Router model."""

    def __init__(
        self,
        catalog: SchemaRouterCatalog,
        *,
        database_id: str,
        instance_id: str,
        sample_rows: int,
        max_sample_chars: int,
    ) -> None:
        self.catalog = catalog
        self.database_id = database_id
        self.instance_id = instance_id
        self.sample_rows = sample_rows
        self.max_sample_chars = max_sample_chars
        self.submission: dict[str, Any] | None = None

    def _family(self, family_id: str) -> TableFamily:
        family = self.catalog.families.get(family_id)
        if family is None or family.database_id != self.database_id:
            raise SelectionValidationError(
                "Family is not present in the allowed database"
            )
        return family

    def list_table_families(
        self, *, cursor: str | None = None, limit: int = 50
    ) -> dict[str, Any]:
        if limit < 1 or limit > 100:
            raise SelectionValidationError("limit must be between 1 and 100")
        if cursor is None:
            offset = 0
        elif cursor.isdigit():
            offset = int(cursor)
        else:
            raise SelectionValidationError("cursor must be an integer string")
        families = self.catalog.database_families(self.database_id)
        page = families[offset : offset + limit]
        return {
            "families": [
                {
                    "family_id": family.family_id,
                    "schema": family.schema_name,
                    "variant_kind": family.variant_kind,
                    "variant_count": len(family.variants),
                    "variant_preview": [
                        variant.value for variant in family.variants[:8]
                    ],
                }
                for family in page
            ],
            "next_cursor": (
                str(offset + limit) if offset + limit < len(families) else None
            ),
        }

    def search_table_candidates(
        self,
        *,
        query: str,
        schemas: list[str] | None = None,
        limit: int = 30,
    ) -> dict[str, Any]:
        tokens = [
            value.lower()
            for value in re.findall(r"[A-Za-z0-9_]+", query)
            if value
        ]
        if not tokens:
            raise SelectionValidationError("query must contain searchable text")
        if limit < 1 or limit > 100:
            raise SelectionValidationError("limit must be between 1 and 100")
        schema_scope = {value.upper() for value in schemas or []}
        scored = []
        for family in self.catalog.database_families(self.database_id):
            if schema_scope and family.schema_name.upper() not in schema_scope:
                continue
            table_text = " ".join(
                [
                    family.family_id,
                    *(variant.table.table_name for variant in family.variants),
                ]
            ).lower()
            column_text = " ".join(
                [
                    *family.common_columns,
                    *family.differing_columns,
                    *(
                        description
                        for variant in family.variants[:2]
                        for description in variant.table.column_descriptions
                    ),
                    *(
                        variant.table.table_description
                        for variant in family.variants[:2]
                    ),
                ]
            ).lower()
            score = sum(
                (5 if token in table_text else 0)
                + (1 if token in column_text else 0)
                for token in tokens
            )
            if score:
                scored.append((score, family))
        scored.sort(key=lambda item: (-item[0], item[1].family_id))
        return {
            "matches": [
                {
                    "family_id": family.family_id,
                    "score": score,
                    "variant_kind": family.variant_kind,
                    "variant_count": len(family.variants),
                    "common_columns": list(family.common_columns[:30]),
                    "differing_columns": list(family.differing_columns[:20]),
                }
                for score, family in scored[:limit]
            ]
        }

    def describe_table_family(
        self,
        *,
        family_id: str,
        variant: str | None = None,
        include_samples: bool = True,
    ) -> dict[str, Any]:
        family = self._family(family_id)
        selected = family.variants[0]
        if variant is not None:
            matches = [value for value in family.variants if value.value == variant]
            if not matches:
                raise SelectionValidationError(
                    "variant is not present in the selected family"
                )
            selected = matches[0]
        table = selected.table
        samples = []
        sample_truncations = []
        if include_samples:
            for index, row in enumerate(table.sample_rows[: self.sample_rows]):
                text, truncated = _bounded(
                    json.dumps(row, ensure_ascii=False, default=str),
                    self.max_sample_chars,
                )
                try:
                    samples.append(json.loads(text) if not truncated else text)
                except json.JSONDecodeError:
                    samples.append(text)
                if truncated:
                    sample_truncations.append(index)
        ddl, ddl_truncated = _bounded(table.ddl, self.max_sample_chars)
        return {
            "family_id": family.family_id,
            "variant_kind": family.variant_kind,
            "variants": [
                {
                    "value": value.value,
                    "table": value.table.full_name,
                }
                for value in family.variants
            ],
            "common_columns": list(family.common_columns[:100]),
            "common_column_count": len(family.common_columns),
            "differing_columns": list(family.differing_columns[:100]),
            "differing_column_count": len(family.differing_columns),
            "representative_similarity_min": family.representative_similarity_min,
            "representative_similarity_median": (
                family.representative_similarity_median
            ),
            "representative_table": table.full_name,
            "description": table.table_description,
            "ddl": ddl,
            "ddl_truncated": ddl_truncated,
            "sample_rows": samples,
            "sample_truncations": sample_truncations,
        }

    def _resolve_selector(
        self, family: TableFamily, selector: dict[str, Any]
    ) -> list[str]:
        if not isinstance(selector, dict):
            raise SelectionValidationError("variant_selector must be an object")
        mode = selector.get("mode")
        if mode not in SELECTOR_MODES:
            raise SelectionValidationError(
                f"variant_selector.mode must be one of {sorted(SELECTOR_MODES)}"
            )
        if mode == "all":
            return list(family.physical_tables)
        if mode == "exact":
            tables = selector.get("tables")
            if not isinstance(tables, list) or not tables:
                raise SelectionValidationError(
                    "exact selector requires a non-empty tables array"
                )
            allowed = set(family.physical_tables)
            selected = []
            for value in tables:
                canonical = self.catalog.canonical_table(str(value))
                if canonical is None or canonical not in allowed:
                    raise SelectionValidationError(
                        f"Table is not a variant of {family.family_id}: {value}"
                    )
                if canonical not in selected:
                    selected.append(canonical)
            return selected
        if mode == "versions":
            values = selector.get("values")
            if not isinstance(values, list) or not values:
                raise SelectionValidationError(
                    "versions selector requires a non-empty values array"
                )
            requested = {str(value) for value in values}
            selected = [
                variant.table.full_name
                for variant in family.variants
                if variant.value in requested
            ]
            missing = requested - {variant.value for variant in family.variants}
            if missing:
                raise SelectionValidationError(
                    f"Unknown family variant(s): {', '.join(sorted(missing))}"
                )
            return selected
        if family.variant_kind != "date":
            raise SelectionValidationError(
                "date_ranges is only valid for YYYYMMDD table families"
            )
        ranges = selector.get("ranges")
        if not isinstance(ranges, list) or not ranges:
            raise SelectionValidationError(
                "date_ranges selector requires a non-empty ranges array"
            )
        selected = []
        for value in ranges:
            if not isinstance(value, dict):
                raise SelectionValidationError("Every date range must be an object")
            start = str(value.get("start", ""))
            end = str(value.get("end", ""))
            if not re.fullmatch(r"\d{8}", start) or not re.fullmatch(r"\d{8}", end):
                raise SelectionValidationError(
                    "Date range start and end must use YYYYMMDD"
                )
            if start > end:
                raise SelectionValidationError(
                    "Date range start must not be after end"
                )
            selected.extend(
                variant.table.full_name
                for variant in family.variants
                if start <= variant.value <= end
            )
        selected = list(dict.fromkeys(selected))
        if not selected:
            raise SelectionValidationError(
                "Date ranges did not match any physical table"
            )
        return selected

    def resolve_family_variants(
        self, *, family_id: str, variant_selector: dict[str, Any]
    ) -> dict[str, Any]:
        family = self._family(family_id)
        tables = self._resolve_selector(family, variant_selector)
        return {
            "family_id": family_id,
            "matched_tables": len(tables),
            "tables": tables[:30],
            "tables_truncated": len(tables) > 30,
        }

    def validate_submission(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise SelectionValidationError("Submission must be an object")
        if payload.get("instance_id") != self.instance_id:
            raise SelectionValidationError(
                "instance_id does not match the current task"
            )
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise SelectionValidationError("candidates must be a non-empty array")
        available = self.catalog.database_families(self.database_id)
        if len(candidates) > len(available):
            raise SelectionValidationError(
                "Candidate count exceeds the allowed database family count"
            )
        seen: set[str] = set()
        normalized = []
        all_tables = []
        for index, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                raise SelectionValidationError("Every candidate must be an object")
            if candidate.get("rank") != index:
                raise SelectionValidationError(
                    "Candidate ranks must be contiguous and start at 1"
                )
            family_id = candidate.get("family_id")
            if not isinstance(family_id, str) or family_id in seen:
                raise SelectionValidationError(
                    "Candidate family ids must be unique strings"
                )
            seen.add(family_id)
            family = self._family(family_id)
            tier = candidate.get("tier")
            if tier not in TIERS:
                raise SelectionValidationError(
                    f"tier must be one of {sorted(TIERS)}"
                )
            roles = candidate.get("roles")
            if (
                not isinstance(roles, list)
                or not roles
                or any(role not in ROLES for role in roles)
            ):
                raise SelectionValidationError(
                    f"roles must contain values from {sorted(ROLES)}"
                )
            reason = candidate.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise SelectionValidationError("reason must be non-empty")
            tables = self._resolve_selector(
                family, candidate.get("variant_selector")
            )
            all_tables.extend(tables)
            normalized.append(
                {
                    "rank": index,
                    "family_id": family_id,
                    "tier": tier,
                    "roles": list(dict.fromkeys(roles)),
                    "reason": reason.strip(),
                    "variant_selector": candidate["variant_selector"],
                    "resolved_physical_tables": tables,
                }
            )
        result = {
            "instance_id": self.instance_id,
            "database_id": self.database_id,
            "candidates": normalized,
            "resolved_physical_tables": list(dict.fromkeys(all_tables)),
            "valid": True,
        }
        self.submission = result
        return result

    def submit_table_selection(
        self, *, candidates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        result = self.validate_submission(
            {"instance_id": self.instance_id, "candidates": candidates}
        )
        return {
            "accepted": True,
            "candidate_families": len(result["candidates"]),
            "resolved_physical_tables": len(
                result["resolved_physical_tables"]
            ),
        }

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "list_table_families": self.list_table_families,
            "search_table_candidates": self.search_table_candidates,
            "describe_table_family": self.describe_table_family,
            "resolve_family_variants": self.resolve_family_variants,
            "submit_table_selection": self.submit_table_selection,
        }
        handler = handlers.get(name)
        if handler is None:
            raise SelectionValidationError(f"Unknown Router tool: {name}")
        return handler(**arguments)


def _object_schema(
    properties: dict[str, Any], required: list[str] | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        result["required"] = required
    return result


VARIANT_SELECTOR_SCHEMA = _object_schema(
    {
        "mode": {"type": "string", "enum": sorted(SELECTOR_MODES)},
        "tables": {"type": "array", "items": {"type": "string"}},
        "values": {"type": "array", "items": {"type": "string"}},
        "ranges": {
            "type": "array",
            "items": _object_schema(
                {
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                },
                ["start", "end"],
            ),
        },
    },
    ["mode"],
)
CANDIDATE_SCHEMA = _object_schema(
    {
        "rank": {"type": "integer", "minimum": 1},
        "family_id": {"type": "string"},
        "tier": {"type": "string", "enum": sorted(TIERS)},
        "roles": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(ROLES)},
            "minItems": 1,
        },
        "reason": {"type": "string"},
        "variant_selector": VARIANT_SELECTOR_SCHEMA,
    },
    ["rank", "family_id", "tier", "roles", "reason", "variant_selector"],
)


EXPLORATION_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_table_families",
            "description": "List logical table families in the allowed database.",
            "parameters": _object_schema(
                {
                    "cursor": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                }
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_table_candidates",
            "description": (
                "Search table families by table names, columns, descriptions, "
                "and schemas in the allowed database."
            ),
            "parameters": _object_schema(
                {
                    "query": {"type": "string"},
                    "schemas": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                ["query"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table_family",
            "description": (
                "Inspect variants, common and differing columns, DDL, and a "
                "bounded local sample for one table family."
            ),
            "parameters": _object_schema(
                {
                    "family_id": {"type": "string"},
                    "variant": {"type": "string"},
                    "include_samples": {"type": "boolean"},
                },
                ["family_id"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_family_variants",
            "description": (
                "Validate a family variant selector and preview matching "
                "physical tables."
            ),
            "parameters": _object_schema(
                {
                    "family_id": {"type": "string"},
                    "variant_selector": VARIANT_SELECTOR_SCHEMA,
                },
                ["family_id", "variant_selector"],
            ),
        },
    },
]
SUBMIT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_table_selection",
        "description": "Submit the final ranked table-family selection.",
        "parameters": _object_schema(
            {
                "candidates": {
                    "type": "array",
                    "items": CANDIDATE_SCHEMA,
                    "minItems": 1,
                },
            },
            ["candidates"],
        ),
    },
}
EXPLORATION_TOOL_SCHEMAS.append(SUBMIT_TOOL_SCHEMA)
SUBMISSION_TOOL_SCHEMAS = [SUBMIT_TOOL_SCHEMA]


def build_router_user_prompt(
    item: dict[str, Any],
    *,
    external_knowledge: str | None,
    family_overview: str,
) -> str:
    return f"""Question: {item['instruction']}
External Knowledge:
{external_knowledge if external_knowledge else 'None'}

Allowed database: {item['db_id']}
Complete logical table-family overview:
{family_overview}

Explore only local metadata in this database. Identify every table family needed
by the official-style solution, including fact tables, filters, output fields,
bridge/join tables, geography/crosswalk tables, and reference dimensions.
Select exact dates, years, periods, or versions when the question determines
them. Before the round limit, submit one ranked selection with
submit_table_selection. Never invent a table or access SQL answer labels."""


class SchemaRouterAgent:
    """Bounded tool-using model loop that returns a validated selection."""
    def __init__(
        self,
        *,
        model_client: OpenAI,
        model_config: dict[str, Any],
        system_prompt: str,
        max_rounds: int,
        max_tool_calls: int,
    ) -> None:
        self.model_client = model_client
        self.model_config = model_config
        self.system_prompt = system_prompt
        self.max_rounds = max_rounds
        self.max_tool_calls = max_tool_calls

    def _call_model(
        self,
        messages: list[dict[str, Any]],
        *,
        tool_schemas: list[dict[str, Any]],
    ) -> tuple[Any, int]:
        retry = self.model_config["retry"]
        delay = retry["initial_delay_seconds"]
        attempts = 0
        while attempts < retry["max_attempts"]:
            attempts += 1
            try:
                response = self.model_client.chat.completions.create(
                    model=self.model_config["name"],
                    messages=messages,
                    tools=tool_schemas,
                    tool_choice="auto",
                    temperature=self.model_config["temperature"],
                    top_p=self.model_config["top_p"],
                    max_tokens=self.model_config["max_tokens"],
                    n=1,
                    **build_model_request_kwargs(
                        self.model_config, location="schema_router.model"
                    ),
                )
                return response, attempts
            except Exception as exc:  # noqa: BLE001
                if (
                    attempts >= retry["max_attempts"]
                    or not self._is_retryable_model_error(exc)
                ):
                    raise
                time.sleep(delay)
                delay = min(
                    delay * retry["backoff_multiplier"],
                    retry["max_delay_seconds"],
                )
        raise SchemaRouterError("Unexpected model retry termination")

    @staticmethod
    def _is_retryable_model_error(exc: Exception) -> bool:
        if isinstance(
            exc,
            (APIConnectionError, APITimeoutError, ConnectionError, TimeoutError),
        ):
            return True
        status_code = getattr(exc, "status_code", None)
        return isinstance(status_code, int) and (
            status_code in {408, 409, 429} or status_code >= 500
        )

    @staticmethod
    def _usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        return {
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "reasoning_tokens": int(getattr(details, "reasoning_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }

    def run(
        self,
        *,
        item: dict[str, Any],
        rollout_idx: int,
        tools: SchemaRouterTools,
        user_prompt: str,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        performance = {
            "model_calls": 0,
            "model_attempts": 0,
            "tool_calls": 0,
            "exploration_tool_calls": 0,
            "tool_errors": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "forced_submissions": 0,
            "format_repairs": 0,
        }
        trace = []
        last_submission_error: str | None = None
        for round_number in range(1, self.max_rounds + 1):
            forced = round_number == self.max_rounds
            stage = "submission_only" if forced else "exploration"
            tool_schemas = (
                SUBMISSION_TOOL_SCHEMAS
                if forced
                else EXPLORATION_TOOL_SCHEMAS
            )
            if forced:
                performance["forced_submissions"] += 1
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Final round: call submit_table_selection now. "
                            "Do not continue exploration or answer with plain text."
                        ),
                    }
                )
            response, attempts = self._call_model(
                messages, tool_schemas=tool_schemas
            )
            performance["model_calls"] += 1
            performance["model_attempts"] += attempts
            usage = self._usage(response)
            for key, value in usage.items():
                performance[key] += value
            message = response.choices[0].message
            wire: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
            }
            raw_calls = message.tool_calls or []
            if raw_calls:
                wire["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in raw_calls
                ]
            reasoning_content = extract_reasoning_content(message, preserve=True)
            if (
                raw_calls
                and reasoning_content
                and deepseek_thinking_enabled(self.model_config)
            ):
                wire["reasoning_content"] = reasoning_content
            messages.append(wire)
            round_trace = {
                "round": round_number,
                "stage": stage,
                "available_tools": [
                    schema["function"]["name"] for schema in tool_schemas
                ],
                "forced_submit": forced,
                "content": message.content or "",
                "tools": [],
            }
            trace.append(round_trace)
            if not raw_calls:
                if forced:
                    last_submission_error = (
                        "Submission-only round did not call "
                        "submit_table_selection"
                    )
                    continue
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Continue metadata exploration or submit the final "
                            "selection with submit_table_selection."
                        ),
                    }
                )
                continue
            for call in raw_calls:
                performance["tool_calls"] += 1
                is_submit = call.function.name == "submit_table_selection"
                if not is_submit:
                    performance["exploration_tool_calls"] += 1
                if (
                    not is_submit
                    and performance["exploration_tool_calls"]
                    > self.max_tool_calls
                ):
                    result = {
                        "error": (
                            "Router tool-call budget exhausted; submit the final "
                            "selection now."
                        )
                    }
                    performance["tool_errors"] += 1
                else:
                    try:
                        arguments = json.loads(call.function.arguments or "{}")
                        if not isinstance(arguments, dict):
                            raise SelectionValidationError(
                                "Tool arguments must be an object"
                            )
                        result = tools.execute(call.function.name, arguments)
                    except Exception as exc:  # noqa: BLE001
                        result = {
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        }
                        performance["tool_errors"] += 1
                        if call.function.name == "submit_table_selection":
                            last_submission_error = str(exc)
                round_trace["tools"].append(
                    {"name": call.function.name, "result": result}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": _json(result),
                    }
                )
                if tools.submission is not None:
                    performance["duration_seconds"] = round(
                        time.perf_counter() - started, 6
                    )
                    return {
                        "instance_id": item["instance_id"],
                        "rollout_idx": rollout_idx,
                        "completed": True,
                        "selection": tools.submission,
                        "trace": trace,
                        "performance": performance,
                    }

        performance["format_repairs"] += 1
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your forced submission was invalid. Repair only the final "
                    "submit_table_selection arguments. Last validation error: "
                    f"{last_submission_error or 'no valid submission was made'}"
                ),
            }
        )
        try:
            response, attempts = self._call_model(
                messages, tool_schemas=SUBMISSION_TOOL_SCHEMAS
            )
            performance["model_calls"] += 1
            performance["model_attempts"] += attempts
            usage = self._usage(response)
            for key, value in usage.items():
                performance[key] += value
            message = response.choices[0].message
            repair_trace = {
                "round": self.max_rounds + 1,
                "stage": "format_repair",
                "available_tools": ["submit_table_selection"],
                "forced_submit": True,
                "format_repair": True,
                "content": message.content or "",
                "tools": [],
            }
            trace.append(repair_trace)
            repair_calls = message.tool_calls or []
            if not repair_calls:
                last_submission_error = (
                    "Format-repair round did not call submit_table_selection"
                )
            for call in repair_calls:
                performance["tool_calls"] += 1
                if call.function.name != "submit_table_selection":
                    continue
                arguments = json.loads(call.function.arguments or "{}")
                tool_result = tools.execute(call.function.name, arguments)
                repair_trace["tools"].append(
                    {"name": call.function.name, "result": tool_result}
                )
                if tools.submission is not None:
                    performance["duration_seconds"] = round(
                        time.perf_counter() - started, 6
                    )
                    return {
                        "instance_id": item["instance_id"],
                        "rollout_idx": rollout_idx,
                        "completed": True,
                        "selection": tools.submission,
                        "trace": trace,
                        "performance": performance,
                    }
        except Exception as exc:  # noqa: BLE001
            last_submission_error = str(exc)
        performance["duration_seconds"] = round(
            time.perf_counter() - started, 6
        )
        return {
            "instance_id": item["instance_id"],
            "rollout_idx": rollout_idx,
            "completed": False,
            "error": last_submission_error or "No valid selection was submitted",
            "trace": trace,
            "performance": performance,
        }


def load_external_knowledge(
    item: dict[str, Any], documents_path: Path
) -> str | None:
    filename = item.get("external_knowledge")
    if not filename:
        return None
    path = documents_path / str(filename)
    if not path.is_file():
        raise SchemaRouterError(
            f"External knowledge does not exist for {item['instance_id']}: {path}"
        )
    return path.read_text(encoding="utf-8").strip()
