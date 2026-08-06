"""Strict configuration and run-directory handling for the Schema Router."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .model_request import ModelRequestConfigError, validate_thinking_config


SCHEMA_ROUTER_PROTOCOL_VERSION = 3


class SchemaRouterConfigError(ValueError):
    """Raised when the standalone Schema Router configuration is invalid."""


@dataclass(frozen=True)
class SchemaRouterConfig:
    config_path: Path
    repo_root: Path
    raw: dict[str, Any]
    secrets: dict[str, Any]
    paths: dict[str, Path]
    fingerprint: str
    run_dir: Path | None = None

    @property
    def experiment_group_dir(self) -> Path:
        experiment = self.raw["experiment"]
        return (
            self.repo_root / experiment["results_root"] / experiment["name"]
        ).resolve()

    @property
    def experiment_dir(self) -> Path:
        return self.run_dir or self.experiment_group_dir


_TOP_LEVEL = {
    "experiment",
    "secrets_file",
    "paths",
    "schema_router",
    "evaluation",
}
_SECTIONS = {
    "experiment": {"name", "results_root", "resume"},
    "paths": {
        "input_file",
        "databases",
        "documents",
        "official_sql_dir",
        "router_prompt",
    },
    "schema_router": {
        "model",
        "max_rounds",
        "max_tool_calls",
        "num_threads",
        "sample_rows",
        "max_sample_chars",
    },
    "schema_router.model": {
        "name",
        "provider",
        "thinking_level",
        "temperature",
        "top_p",
        "max_tokens",
        "request_timeout_seconds",
        "retry",
    },
    "schema_router.model.retry": {
        "max_attempts",
        "initial_delay_seconds",
        "backoff_multiplier",
        "max_delay_seconds",
    },
    "evaluation": {"rollouts", "thresholds"},
    "evaluation.thresholds": {
        "physical_task_full_coverage",
        "physical_micro_recall",
        "invalid_references",
    },
}
_RUN_DIR_PATTERN = re.compile(r"\d{8}-\d{6}(?:-\d{2})?")
_LEGACY_EVALUATION_FIELDS = {
    "phase",
    "split_seed",
    "development_size",
    "development_rollouts",
    "holdout_rollouts",
}


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise SchemaRouterConfigError(f"{label} does not exist: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SchemaRouterConfigError(f"{label} is not valid YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise SchemaRouterConfigError(f"{label} must contain a mapping")
    return value


def _mapping(parent: dict[str, Any], key: str, location: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise SchemaRouterConfigError(f"{location}.{key} must be a mapping")
    return value


def _reject_unknown(
    value: dict[str, Any], allowed: set[str], location: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise SchemaRouterConfigError(
            f"Unknown field(s) in {location}: {', '.join(unknown)}"
        )


def _require(value: dict[str, Any], required: set[str], location: str) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise SchemaRouterConfigError(
            f"Missing field(s) in {location}: {', '.join(missing)}"
        )


def _positive_int(value: Any, location: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SchemaRouterConfigError(f"{location} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise SchemaRouterConfigError(f"{location} must be at least {minimum}")
    return value


def _number(value: Any, location: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SchemaRouterConfigError(f"{location} must be numeric")
    return float(value)


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaRouterConfigError(f"{location} must be a non-empty string")
    return value.strip()


def _resolve(repo_root: Path, value: Any, location: str) -> Path:
    text = _string(value, location)
    path = Path(text)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _validate(raw: dict[str, Any]) -> None:
    _reject_unknown(raw, _TOP_LEVEL, "config")
    _require(raw, _TOP_LEVEL, "config")
    for section in ("experiment", "paths", "schema_router", "evaluation"):
        section_value = _mapping(raw, section, "config")
        if section == "evaluation":
            legacy = sorted(set(section_value) & _LEGACY_EVALUATION_FIELDS)
            if legacy:
                raise SchemaRouterConfigError(
                    "Legacy development/holdout evaluation field(s) are no "
                    f"longer supported: {', '.join(legacy)}; replace them with "
                    "evaluation.rollouts"
                )
        _reject_unknown(section_value, _SECTIONS[section], section)
        _require(section_value, _SECTIONS[section], section)

    experiment = raw["experiment"]
    _string(experiment["name"], "experiment.name")
    _string(experiment["results_root"], "experiment.results_root")
    if not isinstance(experiment["resume"], bool):
        raise SchemaRouterConfigError("experiment.resume must be boolean")

    router = raw["schema_router"]
    for key in (
        "max_rounds",
        "max_tool_calls",
        "num_threads",
        "sample_rows",
        "max_sample_chars",
    ):
        _positive_int(router[key], f"schema_router.{key}")
    if router["max_rounds"] < 2:
        raise SchemaRouterConfigError("schema_router.max_rounds must be at least 2")

    model = _mapping(router, "model", "schema_router")
    _reject_unknown(model, _SECTIONS["schema_router.model"], "schema_router.model")
    _require(
        model,
        _SECTIONS["schema_router.model"] - {"provider", "thinking_level"},
        "schema_router.model",
    )
    try:
        validate_thinking_config(model, location="schema_router.model")
    except ModelRequestConfigError as exc:
        raise SchemaRouterConfigError(str(exc)) from exc
    _string(model["name"], "schema_router.model.name")
    _number(model["temperature"], "schema_router.model.temperature")
    top_p = _number(model["top_p"], "schema_router.model.top_p")
    if not 0 < top_p <= 1:
        raise SchemaRouterConfigError(
            "schema_router.model.top_p must be greater than 0 and at most 1"
        )
    _positive_int(model["max_tokens"], "schema_router.model.max_tokens")
    _number(
        model["request_timeout_seconds"],
        "schema_router.model.request_timeout_seconds",
    )
    retry = _mapping(model, "retry", "schema_router.model")
    _reject_unknown(
        retry,
        _SECTIONS["schema_router.model.retry"],
        "schema_router.model.retry",
    )
    _require(
        retry,
        _SECTIONS["schema_router.model.retry"],
        "schema_router.model.retry",
    )
    _positive_int(retry["max_attempts"], "schema_router.model.retry.max_attempts")
    for key in (
        "initial_delay_seconds",
        "backoff_multiplier",
        "max_delay_seconds",
    ):
        if _number(retry[key], f"schema_router.model.retry.{key}") < 0:
            raise SchemaRouterConfigError(
                f"schema_router.model.retry.{key} must not be negative"
            )

    evaluation = raw["evaluation"]
    _positive_int(evaluation["rollouts"], "evaluation.rollouts")
    thresholds = _mapping(evaluation, "thresholds", "evaluation")
    _reject_unknown(
        thresholds, _SECTIONS["evaluation.thresholds"], "evaluation.thresholds"
    )
    _require(
        thresholds, _SECTIONS["evaluation.thresholds"], "evaluation.thresholds"
    )
    for key in ("physical_task_full_coverage", "physical_micro_recall"):
        threshold = _number(thresholds[key], f"evaluation.thresholds.{key}")
        if not 0 <= threshold <= 1:
            raise SchemaRouterConfigError(
                f"evaluation.thresholds.{key} must be between 0 and 1"
            )
    _positive_int(
        thresholds["invalid_references"],
        "evaluation.thresholds.invalid_references",
        allow_zero=True,
    )


def load_schema_router_config(
    config_path: str | Path,
) -> SchemaRouterConfig:
    path = Path(config_path).expanduser().resolve()
    raw = _read_yaml(path, "Schema Router configuration")
    _validate(raw)
    repo_root = Path(__file__).resolve().parents[3]
    paths = {
        key: _resolve(repo_root, value, f"paths.{key}")
        for key, value in raw["paths"].items()
    }
    for key, value in paths.items():
        expected_file = key in {"input_file", "router_prompt"}
        if expected_file and not value.is_file():
            raise SchemaRouterConfigError(f"paths.{key} is not a file: {value}")
        if not expected_file and not value.is_dir():
            raise SchemaRouterConfigError(
                f"paths.{key} is not a directory: {value}"
            )

    secrets_path = _resolve(repo_root, raw["secrets_file"], "secrets_file")
    secrets = _read_yaml(secrets_path, "Secrets file")
    _reject_unknown(secrets, {"model_api", "snowflake"}, "secrets")
    model_api = _mapping(secrets, "model_api", "secrets")
    _reject_unknown(model_api, {"base_url", "api_key"}, "secrets.model_api")
    _require(model_api, {"base_url", "api_key"}, "secrets.model_api")
    for key in ("base_url", "api_key"):
        _string(model_api[key], f"secrets.model_api.{key}")

    payload = {
        "config": raw,
        "router_protocol_version": SCHEMA_ROUTER_PROTOCOL_VERSION,
        "router_prompt_sha256": hashlib.sha256(
            paths["router_prompt"].read_bytes()
        ).hexdigest(),
        "model_base_url": model_api["base_url"],
        "model_api_identity": hashlib.sha256(
            model_api["api_key"].encode("utf-8")
        ).hexdigest(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return SchemaRouterConfig(
        config_path=path,
        repo_root=repo_root,
        raw=raw,
        secrets=secrets,
        paths=paths,
        fingerprint=fingerprint,
    )


def bind_schema_router_task_set(
    config: SchemaRouterConfig, task_set: dict[str, Any]
) -> SchemaRouterConfig:
    """Bind label/catalog identity before selecting or creating a run directory."""
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "base_fingerprint": config.fingerprint,
                "task_set": task_set,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return replace(config, fingerprint=fingerprint)


def prepare_schema_router_run(config: SchemaRouterConfig) -> SchemaRouterConfig:
    group_dir = config.experiment_group_dir
    group_dir.mkdir(parents=True, exist_ok=True)
    run_dir: Path | None = None
    if config.raw["experiment"]["resume"]:
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
            if (candidate / "schema-router-summary.json").is_file():
                continue
            manifest_path = candidate / "run-manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("fingerprint") == config.fingerprint:
                run_dir = candidate
                break
    if run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = group_dir / stamp
        suffix = 1
        while run_dir.exists():
            run_dir = group_dir / f"{stamp}-{suffix:02d}"
            suffix += 1
        run_dir.mkdir(parents=True)

    resolved = replace(config, run_dir=run_dir)
    manifest_path = run_dir / "run-manifest.json"
    if not manifest_path.is_file():
        snapshot = copy.deepcopy(config.raw)
        snapshot["paths"] = {key: str(value) for key, value in config.paths.items()}
        snapshot["secrets"] = {
            "model_api": {
                "base_url": config.secrets["model_api"]["base_url"],
                "api_key": "***REDACTED***",
            }
        }
        manifest_path.write_text(
            json.dumps(
                {
                    "fingerprint": config.fingerprint,
                    "router_protocol_version": SCHEMA_ROUTER_PROTOCOL_VERSION,
                    "router_prompt_sha256": hashlib.sha256(
                        config.paths["router_prompt"].read_bytes()
                    ).hexdigest(),
                    "started_at": datetime.now().astimezone().isoformat(),
                    "effective_config": snapshot,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return resolved
