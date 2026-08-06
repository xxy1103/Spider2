"""One-command, YAML-driven launcher for Spider Agent TC."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import requests
import yaml

from config import ConfigError, LoadedConfig, load_config, redacted_effective_config
from safe_logging import RedactingFilter, configured_sensitive_values

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Spider Agent TC from one YAML configuration")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML file")
    return parser.parse_args()


def find_available_port(host: str, preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return int(sock.getsockname()[1])
    raise RuntimeError(f"Unable to allocate a local port on {host}")


def _safe_error(exc: BaseException, config: LoadedConfig | None = None) -> str:
    message = str(exc)
    if config:
        sensitive = [config.secrets["model_api"]["api_key"]]
        if "snowflake" in config.secrets:
            sensitive.extend(
                [
                    config.secrets["snowflake"].get("user"),
                    config.secrets["snowflake"].get("password"),
                ]
            )
        for value in sensitive:
            if value:
                message = message.replace(value, "***REDACTED***")
    return message


def check_snowflake(config: LoadedConfig) -> None:
    if config.raw["tools"]["sql"]["mode"] != "live":
        raise ConfigError("Snowflake connectivity checks are unavailable in mock mode")
    import snowflake.connector

    settings = config.raw["tools"]["sql"]
    connection = snowflake.connector.connect(
        **config.secrets["snowflake"],
        login_timeout=settings["timeout_seconds"],
        network_timeout=settings["timeout_seconds"],
    )
    try:
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_WAREHOUSE()")
            cursor.fetchone()
        finally:
            cursor.close()
    finally:
        connection.close()


def check_dependencies() -> None:
    modules = {
        "fastapi": "fastapi",
        "openai": "openai",
        "pandas": "pandas",
        "PyYAML": "yaml",
        "requests": "requests",
        "rich": "rich",
        "snowflake-connector-python": "snowflake.connector",
        "sqlglot": "sqlglot",
        "uvicorn": "uvicorn",
        "langgraph": "langgraph",
        "langchain-core": "langchain_core",
        "langchain-openai": "langchain_openai",
    }
    missing = []
    for package, module in modules.items():
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError):
            available = False
        if not available:
            missing.append(package)
    if missing:
        raise ConfigError(
            "Missing Python dependencies: "
            + ", ".join(sorted(missing))
            + ". Create/update the Conda environment from environment.yml."
        )


def make_openai_client(config: LoadedConfig):
    from openai import OpenAI

    return OpenAI(
        base_url=config.secrets["model_api"]["base_url"],
        api_key=config.secrets["model_api"]["api_key"],
        timeout=config.raw["model"]["request_timeout_seconds"],
    )


def check_model(config: LoadedConfig) -> None:
    from agent.model_request import build_model_request_kwargs

    client = make_openai_client(config)
    client.chat.completions.create(
        model=config.raw["model"]["name"],
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        temperature=0,
        max_tokens=8,
        n=1,
        **build_model_request_kwargs(config.raw["model"]),
    )


def run_preflight(config: LoadedConfig, port: int) -> None:
    print("预检中...")
    check_dependencies()
    if config.raw["preflight"]["check_snowflake"]:
        check_snowflake(config)
    if config.raw["preflight"]["check_model"]:
        check_model(config)
    print("预检就绪")


def _environment_versions() -> dict[str, str]:
    packages = [
        "openai",
        "requests",
        "rich",
        "fastapi",
        "uvicorn",
        "pandas",
        "snowflake-connector-python",
        "sqlglot",
        "PyYAML",
        "langgraph",
        "langchain-core",
        "langchain-openai",
    ]
    versions = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def prepare_experiment(config: LoadedConfig, port: int) -> None:
    output_dir = config.experiment_dir
    manifest_path = output_dir / "run-manifest.json"
    nonempty = output_dir.exists() and any(output_dir.iterdir())
    resume = config.raw["experiment"]["resume"]

    if nonempty:
        if not resume:
            raise ConfigError(
                f"Experiment directory already exists and resume is false: {output_dir}"
            )
        if not manifest_path.is_file():
            raise ConfigError(f"Existing experiment has no run manifest: {output_dir}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != config.fingerprint:
            raise ConfigError(
                "Existing experiment configuration differs from this run; choose a new experiment.name"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = redacted_effective_config(config, resolved_port=port)
    (output_dir / "effective-config.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (output_dir / "selected-tasks.json").write_text(
        json.dumps(
            {
                "seed": config.raw["tasks"]["seed"],
                "instance_ids": [item["instance_id"] for item in config.selected_items],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        "fingerprint": config.fingerprint,
        "experiment_name": config.raw["experiment"]["name"],
        "run_id": output_dir.name,
        "created_at": (
            json.loads(manifest_path.read_text(encoding="utf-8")).get("created_at")
            if manifest_path.is_file()
            else datetime.now(timezone.utc).isoformat()
        ),
        "last_started_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config.config_path),
        "environment": _environment_versions(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def start_server(config: LoadedConfig, port: int) -> subprocess.Popen:
    command = [
        sys.executable,
        "-m",
        "servers.serve",
        "--config",
        str(config.config_path),
        "--host",
        config.raw["server"]["host"],
        "--port",
        str(port),
        "--run-dir",
        str(config.experiment_dir),
    ]
    log_path = config.experiment_dir / "run.log"
    log_handle = log_path.open("a", encoding="utf-8")
    try:
        return subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parent,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    finally:
        log_handle.close()


def wait_for_server(config: LoadedConfig, port: int, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + config.raw["server"]["startup_timeout_seconds"]
    url = f"http://{config.raw['server']['host']}:{port}/health"
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Tool server exited early with status {process.returncode}")
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(0.2)
    raise RuntimeError(f"Tool server health check timed out: {last_error or url}")


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def build_agent_args(config: LoadedConfig, port: int) -> SimpleNamespace:
    from agent.model_request import build_model_request_kwargs

    raw = config.raw
    return SimpleNamespace(
        input_file=str(config.paths["input_file"]),
        output_folder=str(config.experiment_dir),
        system_prompt_path=str(config.paths["system_prompt"]),
        databases_path=str(config.paths["databases"]),
        documents_path=str(config.paths["documents"]),
        model=raw["model"]["name"],
        temperature=raw["model"]["temperature"],
        top_p=raw["model"]["top_p"],
        max_new_tokens=raw["model"]["max_tokens"],
        model_request_timeout=raw["model"]["request_timeout_seconds"],
        retry=raw["model"]["retry"],
        model_request_kwargs=build_model_request_kwargs(raw["model"]),
        provider=raw["model"].get("provider"),
        thinking_level=raw["model"].get("thinking_level"),
        api_host=raw["server"]["host"],
        api_port=port,
        tool_request_timeout=raw["server"]["request_timeout_seconds"],
        max_rounds=raw["agent"]["max_rounds"],
        num_threads=raw["agent"]["num_threads"],
        rollout_number=raw["agent"]["rollout_number"],
        routing_index_path=str(config.experiment_dir / "routing-index.json"),
        prompt_strategy="spider-agent",
        model_base_url=config.secrets["model_api"]["base_url"],
        model_api_key=config.secrets["model_api"]["api_key"],
    )


def configure_file_logging(config: LoadedConfig) -> Path:
    """Send application diagnostics to the experiment log, not the terminal."""
    log_path = config.experiment_dir / "run.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler.addFilter(RedactingFilter(configured_sensitive_values(config)))
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    return log_path


def write_summary(config: LoadedConfig, summary: dict[str, Any]) -> None:
    summary["snowflake_mode"] = config.raw["tools"]["sql"]["mode"]
    summary["mock_run"] = config.raw["tools"]["sql"]["mode"] == "mock"
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    routing_path = config.experiment_dir / "routing-index.json"
    if routing_path.is_file():
        routing = json.loads(routing_path.read_text(encoding="utf-8"))
        routes = list(routing.get("routes", {}).values())
        summary["schema_router"] = {
            "mode": routing.get("mode"),
            "failure_policy": routing.get("failure_policy"),
            "expected_routes": routing.get("expected_routes", 0),
            "valid_routes": routing.get("valid_routes", 0),
            "failed_routes": routing.get("failed_routes", 0),
            "failed_route_keys": routing.get("failed_route_keys", []),
            "performance": routing.get("performance", {}),
            "average_tables_before": (
                sum(route["available_physical_tables"] for route in routes)
                / len(routes)
                if routes
                else 0
            ),
            "average_tables_after": (
                sum(len(route["allowed_physical_tables"]) for route in routes)
                / len(routes)
                if routes
                else 0
            ),
            "routes": [
                {
                    "instance_id": route["instance_id"],
                    "rollout_idx": route["rollout_idx"],
                    "tables_before": route["available_physical_tables"],
                    "tables_after": len(route["allowed_physical_tables"]),
                }
                for route in routes
            ],
        }
    (config.experiment_dir / "run-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (config.experiment_dir / "failed-tasks.json").write_text(
        json.dumps(summary["failed_instance_ids"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def execute(config: LoadedConfig) -> int:
    port = find_available_port(config.raw["server"]["host"], config.raw["server"]["preferred_port"])
    run_preflight(config, port)
    prepare_experiment(config, port)
    from servers.structured_tools import build_catalog
    from agent.schema_router import SchemaRouterCatalog
    from agent.schema_router_runtime import run_integrated_schema_router

    log_path = configure_file_logging(config)
    build_catalog(config)
    router_catalog = SchemaRouterCatalog(
        config.paths["databases"],
        {item["db_id"] for item in config.selected_items},
    )
    run_integrated_schema_router(config, router_catalog)

    process = None
    try:
        print("工具服务启动中...")
        process = start_server(config, port)
        wait_for_server(config, port, process)
        print("工具服务就绪")
        from agent.main import run_agent

        agent_started_at = time.perf_counter()
        summary = run_agent(build_agent_args(config, port), config.selected_items)
        summary.setdefault("performance", {})["agent_wall_clock_seconds"] = round(
            time.perf_counter() - agent_started_at, 6
        )
        write_summary(config, summary)
        
        # Auto evaluation and report generation
        auto_eval_config = config.raw.get("auto_evaluate", {})
        if auto_eval_config.get("enabled", False):
            try:
                from agent.auto_evaluator import run_evaluation_and_report
                run_evaluation_and_report(config, summary)
            except Exception as e:
                logger.error("Auto evaluation failed: %s", _safe_error(e, config))
                print(f"自动评分失败，详情见 {log_path}", file=sys.stderr)
        
        if summary["failed_instance_ids"]:
            print(f"运行结束，详情见 {log_path}")
            return 1
        print(f"运行成功，详情见 {log_path}")
        return 0
    finally:
        stop_server(process)


def main() -> int:
    config: LoadedConfig | None = None
    try:
        args = parse_args()
        config = load_config(args.config)
        return execute(config)
    except KeyboardInterrupt:
        print("Interrupted; tool server cleanup completed", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {_safe_error(exc, config)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
