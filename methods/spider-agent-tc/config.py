"""Configuration loading, validation, task selection, and safe snapshots."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a run configuration is invalid."""


@dataclass(frozen=True)
class LoadedConfig:
    config_path: Path
    repo_root: Path
    raw: dict[str, Any]
    secrets: dict[str, Any]
    paths: dict[str, Path]
    selected_items: list[dict[str, Any]]
    fingerprint: str
    run_dir: Path | None = None

    @property
    def experiment_group_dir(self) -> Path:
        experiment = self.raw["experiment"]
        return (self.repo_root / experiment["results_root"] / experiment["name"]).resolve()

    @property
    def experiment_dir(self) -> Path:
        return self.run_dir or self.experiment_group_dir


_SCHEMA = {
    "experiment": {"name", "results_root", "resume"},
    "paths": {"input_file", "databases", "documents", "system_prompt"},
    "tasks": {"instance_ids", "index_ranges", "databases", "sample_size", "seed", "order"},
    "model": {"name", "temperature", "top_p", "max_tokens", "request_timeout_seconds", "retry"},
    "model.retry": {
        "max_attempts",
        "initial_delay_seconds",
        "backoff_multiplier",
        "max_delay_seconds",
    },
    "agent": {"max_rounds", "num_threads", "rollout_number"},
    "server": {
        "host",
        "preferred_port",
        "workers_per_tool",
        "startup_timeout_seconds",
        "request_timeout_seconds",
    },
    "tools": {"bash", "snowflake"},
    "tools.bash": {"timeout_seconds", "max_output_chars"},
    "tools.snowflake": {"mode", "timeout_seconds", "max_output_chars", "mock"},
    "tools.snowflake.mock": {"response_csv"},
    "preflight": {"check_model", "check_snowflake"},
    "auto_evaluate": {"enabled", "timeout", "max_workers"},
}

_TOP_LEVEL = {
    "experiment",
    "secrets_file",
    "paths",
    "tasks",
    "model",
    "agent",
    "server",
    "tools",
    "preflight",
    "auto_evaluate",
}


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"{label} does not exist: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{label} is not valid YAML: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{label} must contain a YAML mapping: {path}")
    return data


def _mapping(parent: dict[str, Any], key: str, location: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{location}.{key} must be a mapping")
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"Unknown field(s) in {location}: {', '.join(unknown)}")


def _required(mapping: dict[str, Any], required: set[str], location: str) -> None:
    missing = sorted(required - set(mapping))
    if missing:
        raise ConfigError(f"Missing field(s) in {location}: {', '.join(missing)}")


def _positive_int(value: Any, location: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{location} must be a {qualifier} integer")
    return value


def _positive_number(value: Any, location: str, *, allow_zero: bool = False) -> float:
    minimum_ok = value >= 0 if allow_zero and isinstance(value, (int, float)) else value > 0 if isinstance(value, (int, float)) else False
    if isinstance(value, bool) or not minimum_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{location} must be a {qualifier} number")
    return float(value)


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{location} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, location: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ConfigError(f"{location} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ConfigError(f"{location} contains duplicate values")
    return [item.strip() for item in value]


def _validate_main(raw: dict[str, Any]) -> None:
    _reject_unknown(raw, _TOP_LEVEL, "config")
    # auto_evaluate is optional
    required_sections = _TOP_LEVEL - {"auto_evaluate"}
    _required(raw, required_sections, "config")

    for section in ("experiment", "paths", "tasks", "model", "agent", "server", "tools", "preflight"):
        mapping = _mapping(raw, section, "config")
        _reject_unknown(mapping, _SCHEMA[section], section)
        _required(mapping, _SCHEMA[section], section)

    # Validate auto_evaluate if present
    if "auto_evaluate" in raw:
        auto_eval = _mapping(raw, "auto_evaluate", "config")
        _reject_unknown(auto_eval, _SCHEMA["auto_evaluate"], "auto_evaluate")

    retry = _mapping(raw["model"], "retry", "model")
    _reject_unknown(retry, _SCHEMA["model.retry"], "model.retry")
    _required(retry, _SCHEMA["model.retry"], "model.retry")
    for tool in ("bash",):
        settings = _mapping(raw["tools"], tool, "tools")
        _reject_unknown(settings, _SCHEMA[f"tools.{tool}"], f"tools.{tool}")
        _required(settings, _SCHEMA[f"tools.{tool}"], f"tools.{tool}")
    snowflake = _mapping(raw["tools"], "snowflake", "tools")
    _reject_unknown(snowflake, _SCHEMA["tools.snowflake"], "tools.snowflake")
    _required(
        snowflake,
        {"mode", "timeout_seconds", "max_output_chars"},
        "tools.snowflake",
    )
    if snowflake["mode"] not in {"live", "mock"}:
        raise ConfigError("tools.snowflake.mode must be 'live' or 'mock'")
    if snowflake["mode"] == "mock":
        mock = _mapping(snowflake, "mock", "tools.snowflake")
        _reject_unknown(mock, _SCHEMA["tools.snowflake.mock"], "tools.snowflake.mock")
        _required(mock, _SCHEMA["tools.snowflake.mock"], "tools.snowflake.mock")
        mock["response_csv"] = _string(
            mock["response_csv"], "tools.snowflake.mock.response_csv"
        )
        if raw["preflight"]["check_snowflake"]:
            raise ConfigError(
                "preflight.check_snowflake must be false when tools.snowflake.mode is 'mock'"
            )
    elif "mock" in snowflake:
        raise ConfigError("tools.snowflake.mock is only allowed when mode is 'mock'")

    experiment = raw["experiment"]
    experiment["name"] = _string(experiment["name"], "experiment.name")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", experiment["name"]):
        raise ConfigError("experiment.name may contain only letters, numbers, '.', '_' and '-'")
    experiment["results_root"] = _string(experiment["results_root"], "experiment.results_root")
    if Path(experiment["results_root"]).is_absolute() or ".." in Path(experiment["results_root"]).parts:
        raise ConfigError("experiment.results_root must be a repository-relative path")
    if not isinstance(experiment["resume"], bool):
        raise ConfigError("experiment.resume must be true or false")

    raw["secrets_file"] = _string(raw["secrets_file"], "secrets_file")
    for key, value in raw["paths"].items():
        raw["paths"][key] = _string(value, f"paths.{key}")

    tasks = raw["tasks"]
    tasks["instance_ids"] = _string_list(tasks["instance_ids"], "tasks.instance_ids")
    tasks["index_ranges"] = _string_list(tasks["index_ranges"], "tasks.index_ranges")
    tasks["databases"] = _string_list(tasks["databases"], "tasks.databases")
    if tasks["sample_size"] is not None:
        _positive_int(tasks["sample_size"], "tasks.sample_size")
    _positive_int(tasks["seed"], "tasks.seed", allow_zero=True)
    if tasks["order"] != "seeded_shuffle":
        raise ConfigError("tasks.order must be 'seeded_shuffle'")

    model = raw["model"]
    model["name"] = _string(model["name"], "model.name")
    if "<" in model["name"] or ">" in model["name"] or model["name"].startswith("replace-"):
        raise ConfigError("model.name still contains a placeholder")
    _positive_number(model["temperature"], "model.temperature", allow_zero=True)
    top_p = _positive_number(model["top_p"], "model.top_p")
    if top_p > 1:
        raise ConfigError("model.top_p must be at most 1")
    _positive_int(model["max_tokens"], "model.max_tokens")
    _positive_number(model["request_timeout_seconds"], "model.request_timeout_seconds")
    _positive_int(retry["max_attempts"], "model.retry.max_attempts")
    _positive_number(retry["initial_delay_seconds"], "model.retry.initial_delay_seconds", allow_zero=True)
    _positive_number(retry["backoff_multiplier"], "model.retry.backoff_multiplier")
    _positive_number(retry["max_delay_seconds"], "model.retry.max_delay_seconds", allow_zero=True)

    for key in ("max_rounds", "num_threads", "rollout_number"):
        _positive_int(raw["agent"][key], f"agent.{key}")
    server = raw["server"]
    server["host"] = _string(server["host"], "server.host")
    port = _positive_int(server["preferred_port"], "server.preferred_port")
    if port > 65535:
        raise ConfigError("server.preferred_port must be at most 65535")
    _positive_int(server["workers_per_tool"], "server.workers_per_tool")
    for key in ("startup_timeout_seconds", "request_timeout_seconds"):
        _positive_number(server[key], f"server.{key}")
    for tool in ("bash", "snowflake"):
        _positive_number(raw["tools"][tool]["timeout_seconds"], f"tools.{tool}.timeout_seconds")
        _positive_int(raw["tools"][tool]["max_output_chars"], f"tools.{tool}.max_output_chars")
    for key in ("check_model", "check_snowflake"):
        if not isinstance(raw["preflight"][key], bool):
            raise ConfigError(f"preflight.{key} must be true or false")


def _validate_secrets(secrets: dict[str, Any], snowflake_mode: str) -> None:
    allowed = {"model_api", "snowflake"}
    _reject_unknown(secrets, allowed, "secrets")
    _required(secrets, {"model_api"}, "secrets")
    model_api = _mapping(secrets, "model_api", "secrets")
    _reject_unknown(model_api, {"base_url", "api_key"}, "secrets.model_api")
    _required(model_api, {"base_url", "api_key"}, "secrets.model_api")
    for key in ("base_url", "api_key"):
        model_api[key] = _string(model_api[key], f"secrets.model_api.{key}")

    snow_fields = {"user", "password", "account", "role", "warehouse"}
    snowflake = secrets.get("snowflake")
    if snowflake_mode == "live":
        snowflake = _mapping(secrets, "snowflake", "secrets")
        _reject_unknown(snowflake, snow_fields, "secrets.snowflake")
        _required(snowflake, snow_fields, "secrets.snowflake")
        for key in snow_fields:
            snowflake[key] = _string(snowflake[key], f"secrets.snowflake.{key}")
    elif snowflake is not None:
        if not isinstance(snowflake, dict):
            raise ConfigError("secrets.snowflake must be a mapping")
        _reject_unknown(snowflake, snow_fields, "secrets.snowflake")
        for key, value in snowflake.items():
            snowflake[key] = _string(value, f"secrets.snowflake.{key}")

    placeholders = {"secret", "username", "python-token", "https://example.com/v1"}
    values = list(model_api.values()) + list((snowflake or {}).values())
    if any(value.strip().lower() in placeholders for value in values):
        raise ConfigError("secrets file still contains example placeholder values")


def _resolve_repo_path(repo_root: Path, value: str, location: str) -> Path:
    candidate = (repo_root / value).resolve()
    try:
        candidate.relative_to(repo_root)
    except ValueError as exc:
        raise ConfigError(f"{location} must remain inside the repository: {value}") from exc
    return candidate


def _parse_ranges(ranges: list[str], item_count: int) -> set[int]:
    selected: set[int] = set()
    for value in ranges:
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", value)
        if not match:
            raise ConfigError(f"Invalid tasks.index_ranges value: {value!r}; use 'N' or 'N-M'")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start > end or end >= item_count:
            raise ConfigError(f"tasks.index_ranges value is outside 0-{item_count - 1}: {value!r}")
        selected.update(range(start, end + 1))
    return selected


def select_tasks(items: list[dict[str, Any]], tasks: dict[str, Any]) -> list[dict[str, Any]]:
    if not items:
        raise ConfigError("Input JSONL contains no tasks")
    for index, item in enumerate(items):
        for field in ("instance_id", "db_id", "instruction"):
            if not isinstance(item.get(field), str) or not item[field]:
                raise ConfigError(f"Input task at line {index + 1} has no valid {field!r}")

    candidates = list(enumerate(items))
    requested_ids = set(tasks["instance_ids"])
    if requested_ids:
        available_ids = {item["instance_id"] for item in items}
        missing = sorted(requested_ids - available_ids)
        if missing:
            raise ConfigError(f"Unknown tasks.instance_ids: {', '.join(missing)}")
        candidates = [(index, item) for index, item in candidates if item["instance_id"] in requested_ids]

    if tasks["index_ranges"]:
        requested_indices = _parse_ranges(tasks["index_ranges"], len(items))
        candidates = [(index, item) for index, item in candidates if index in requested_indices]

    requested_databases = set(tasks["databases"])
    if requested_databases:
        available_databases = {item["db_id"] for item in items}
        missing = sorted(requested_databases - available_databases)
        if missing:
            raise ConfigError(f"Unknown tasks.databases: {', '.join(missing)}")
        candidates = [(index, item) for index, item in candidates if item["db_id"] in requested_databases]

    if not candidates:
        raise ConfigError("Task filters selected zero items")

    rng = random.Random(tasks["seed"])
    sample_size = tasks["sample_size"]
    if sample_size is not None:
        if sample_size > len(candidates):
            raise ConfigError(
                f"tasks.sample_size ({sample_size}) exceeds filtered task count ({len(candidates)})"
            )
        candidates = rng.sample(candidates, sample_size)
    rng.shuffle(candidates)
    return [item for _, item in candidates]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    items = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ConfigError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ConfigError(f"Task at {path}:{line_number} must be an object")
                items.append(value)
    except OSError as exc:
        raise ConfigError(f"Unable to read input file {path}: {exc}") from exc
    return items


def redacted_effective_config(config: LoadedConfig, *, resolved_port: int | None = None) -> dict[str, Any]:
    snapshot = copy.deepcopy(config.raw)
    snapshot["paths"] = {key: str(value) for key, value in config.paths.items()}
    snapshot["secrets"] = {
        "model_api": {
            "base_url": config.secrets["model_api"]["base_url"],
            "api_key": "***REDACTED***",
        },
    }
    if "snowflake" in config.secrets:
        snapshot["secrets"]["snowflake"] = {
            key: "***REDACTED***" if key in {"user", "password"} else value
            for key, value in config.secrets["snowflake"].items()
        }
    if resolved_port is not None:
        snapshot["server"]["resolved_port"] = resolved_port
    return snapshot


_RUN_DIR_PATTERN = re.compile(r"\d{8}-\d{6}(?:-\d{2})?")


def select_experiment_run_dir(
    group_dir: Path,
    *,
    resume: bool,
    fingerprint: str,
    now: datetime | None = None,
) -> Path:
    """Select an interrupted matching run or allocate a new timestamped run."""
    if resume and group_dir.is_dir():
        candidates = sorted(
            (
                path
                for path in group_dir.iterdir()
                if path.is_dir() and _RUN_DIR_PATTERN.fullmatch(path.name)
            ),
            key=lambda path: path.name,
            reverse=True,
        )
        for candidate in candidates:
            if (candidate / "run-summary.json").is_file():
                continue
            manifest_path = candidate / "run-manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("fingerprint") == fingerprint:
                return candidate

    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    candidate = group_dir / timestamp
    suffix = 1
    while candidate.exists():
        candidate = group_dir / f"{timestamp}-{suffix:02d}"
        suffix += 1
    return candidate


def load_config(config_path: str | Path) -> LoadedConfig:
    path = Path(config_path).expanduser().resolve()
    raw = _read_yaml(path, "Configuration file")
    _validate_main(raw)

    repo_root = Path(__file__).resolve().parents[2]
    paths = {
        key: _resolve_repo_path(repo_root, value, f"paths.{key}")
        for key, value in raw["paths"].items()
    }
    for key, value in paths.items():
        expected_file = key in {"input_file", "system_prompt"}
        if expected_file and not value.is_file():
            raise ConfigError(f"paths.{key} is not a file: {value}")
        if not expected_file and not value.is_dir():
            raise ConfigError(f"paths.{key} is not a directory: {value}")

    secrets_path = _resolve_repo_path(repo_root, raw["secrets_file"], "secrets_file")
    secrets = _read_yaml(secrets_path, "Secrets file")
    snowflake_mode = raw["tools"]["snowflake"]["mode"]
    _validate_secrets(secrets, snowflake_mode)

    items = _load_jsonl(paths["input_file"])
    selected = select_tasks(items, raw["tasks"])
    fingerprint_payload = {
        "config": raw,
        "selected_instance_ids": [item["instance_id"] for item in selected],
        "connection_context": {
            "model_base_url": secrets["model_api"]["base_url"],
            "snowflake_mode": snowflake_mode,
            "snowflake_account": secrets.get("snowflake", {}).get("account"),
            "snowflake_role": secrets.get("snowflake", {}).get("role"),
            "snowflake_warehouse": secrets.get("snowflake", {}).get("warehouse"),
            "model_api_identity": hashlib.sha256(
                secrets["model_api"]["api_key"].encode("utf-8")
            ).hexdigest(),
            "snowflake_user_identity": (
                hashlib.sha256(secrets["snowflake"]["user"].encode("utf-8")).hexdigest()
                if snowflake_mode == "live"
                else None
            ),
            "snowflake_token_identity": (
                hashlib.sha256(
                    secrets["snowflake"]["password"].encode("utf-8")
                ).hexdigest()
                if snowflake_mode == "live"
                else None
            ),
        },
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    group_dir = (
        repo_root / raw["experiment"]["results_root"] / raw["experiment"]["name"]
    ).resolve()
    run_dir = select_experiment_run_dir(
        group_dir,
        resume=raw["experiment"]["resume"],
        fingerprint=fingerprint,
    )
    return LoadedConfig(
        config_path=path,
        repo_root=repo_root,
        raw=raw,
        secrets=secrets,
        paths=paths,
        selected_items=selected,
        fingerprint=fingerprint,
        run_dir=run_dir,
    )
