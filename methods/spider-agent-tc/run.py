"""One-command, YAML-driven launcher for Spider Agent TC."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
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
    if config.raw["tools"]["snowflake"]["mode"] != "live":
        raise ConfigError("Snowflake connectivity checks are unavailable in mock mode")
    import snowflake.connector

    settings = config.raw["tools"]["snowflake"]
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
        "snowflake-connector-python": "snowflake.connector",
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
    client = make_openai_client(config)
    client.chat.completions.create(
        model=config.raw["model"]["name"],
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        temperature=0,
        max_tokens=8,
        n=1,
    )


def run_preflight(config: LoadedConfig, port: int) -> None:
    print(f"Selected {len(config.selected_items)} task(s)")
    snowflake_mode = config.raw["tools"]["snowflake"]["mode"]
    print(f"Snowflake tool mode: {snowflake_mode.upper()}")
    print(f"Tool server will use {config.raw['server']['host']}:{port}")
    print("Checking Python dependencies...")
    check_dependencies()
    print("Python dependencies are ready")
    if config.raw["preflight"]["check_snowflake"]:
        print("Checking Snowflake connection...")
        check_snowflake(config)
        print("Snowflake connection is ready")
    if config.raw["preflight"]["check_model"]:
        print("Checking model connection...")
        check_model(config)
        print("Model connection is ready")


def _environment_versions() -> dict[str, str]:
    packages = [
        "openai",
        "requests",
        "fastapi",
        "uvicorn",
        "pandas",
        "snowflake-connector-python",
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
    ]
    return subprocess.Popen(command, cwd=Path(__file__).resolve().parent)


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
        api_host=raw["server"]["host"],
        api_port=port,
        tool_request_timeout=raw["server"]["request_timeout_seconds"],
        max_rounds=raw["agent"]["max_rounds"],
        num_threads=raw["agent"]["num_threads"],
        rollout_number=raw["agent"]["rollout_number"],
        prompt_strategy="spider-agent",
        model_base_url=config.secrets["model_api"]["base_url"],
        model_api_key=config.secrets["model_api"]["api_key"],
    )


def write_summary(config: LoadedConfig, summary: dict[str, Any]) -> None:
    summary["snowflake_mode"] = config.raw["tools"]["snowflake"]["mode"]
    summary["mock_run"] = config.raw["tools"]["snowflake"]["mode"] == "mock"
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
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

    process = None
    try:
        process = start_server(config, port)
        wait_for_server(config, port, process)
        from agent.main import run_agent

        summary = run_agent(build_agent_args(config, port), config.selected_items)
        write_summary(config, summary)
        
        # Auto evaluation and report generation
        auto_eval_config = config.raw.get("auto_evaluate", {})
        if auto_eval_config.get("enabled", False):
            try:
                from agent.auto_evaluator import run_evaluation_and_report
                run_evaluation_and_report(config, summary)
            except Exception as e:
                print(f"\n⚠️  Warning: Auto evaluation failed: {e}", file=sys.stderr)
                print("Agent run results are still valid.\n", file=sys.stderr)
        
        if summary["failed_instance_ids"]:
            print(f"Run completed with {len(summary['failed_instance_ids'])} failed task(s)")
            return 1
        print("Run completed successfully")
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
