import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.schema_router import (
    SchemaRouterAgent,
    SchemaRouterCatalog,
    SchemaRouterTools,
    SelectionValidationError,
    build_router_user_prompt,
)
from agent.schema_router_evaluator import (
    aggregate_scores,
    extract_official_sql_labels,
    make_task_set,
    render_report,
    score_rollout,
    threshold_status,
)


def _write_table(
    root: Path,
    database: str,
    schema: str,
    table: str,
    columns=("id", "value"),
):
    path = root / database / schema / f"{table}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "table_name": f"{schema}.{table}",
                "table_fullname": f"{database}.{schema}.{table}",
                "column_names": list(columns),
                "column_types": ["NUMBER"] * len(columns),
                "description": [""] * len(columns),
                "sample_rows": [{"id": 1, "value": "sample"}],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def catalog(tmp_path):
    databases = tmp_path / "databases"
    for day in ("20170701", "20170702", "20170703"):
        _write_table(
            databases,
            "GA360",
            "GOOGLE_ANALYTICS_SAMPLE",
            f"GA_SESSIONS_{day}",
        )
    _write_table(databases, "GA360", "OTHER_SCHEMA", "LOOKUP_20170701")
    for version in ("23", "24", "29"):
        _write_table(
            databases,
            "EBI_CHEMBL",
            "EBI_CHEMBL",
            f"ACTIVITIES_{version}",
            columns=("activity_id", "molregno"),
        )
    _write_table(
        databases,
        "EBI_CHEMBL",
        "EBI_CHEMBL",
        "ACTIVITIES",
        columns=("activity_id", "molregno", "new_column"),
    )
    return SchemaRouterCatalog(databases)


def test_family_catalog_groups_partitions_and_versions(catalog):
    ga_family = catalog.families[
        "GA360.GOOGLE_ANALYTICS_SAMPLE.GA_SESSIONS_{YYYYMMDD}"
    ]
    assert ga_family.variant_kind == "date"
    assert [variant.value for variant in ga_family.variants] == [
        "20170701",
        "20170702",
        "20170703",
    ]

    chembl = catalog.families[
        "EBI_CHEMBL.EBI_CHEMBL.ACTIVITIES_{VERSION}"
    ]
    assert chembl.variant_kind == "version"
    assert {variant.value for variant in chembl.variants} == {
        "23",
        "24",
        "29",
        "current",
    }
    assert (
        "GA360.OTHER_SCHEMA.LOOKUP_{YYYYMMDD}" in catalog.families
    )
    assert (
        catalog.table_to_family[
            "GA360.OTHER_SCHEMA.LOOKUP_20170701"
        ]
        != ga_family.family_id
    )


def test_selection_resolves_date_range_and_rejects_invalid_rank(catalog):
    tools = SchemaRouterTools(
        catalog,
        database_id="GA360",
        instance_id="sf_bq010",
        sample_rows=1,
        max_sample_chars=1000,
    )
    candidate = {
        "rank": 1,
        "family_id": (
            "GA360.GOOGLE_ANALYTICS_SAMPLE.GA_SESSIONS_{YYYYMMDD}"
        ),
        "tier": "required",
        "roles": ["fact", "filter"],
        "reason": "Sessions in the requested interval",
        "variant_selector": {
            "mode": "date_ranges",
            "ranges": [{"start": "20170702", "end": "20170703"}],
        },
    }
    result = tools.validate_submission(
        {"instance_id": "sf_bq010", "candidates": [candidate]}
    )
    assert result["resolved_physical_tables"] == [
        "GA360.GOOGLE_ANALYTICS_SAMPLE.GA_SESSIONS_20170702",
        "GA360.GOOGLE_ANALYTICS_SAMPLE.GA_SESSIONS_20170703",
    ]

    invalid = dict(candidate, rank=2)
    with pytest.raises(SelectionValidationError, match="contiguous"):
        tools.validate_submission(
            {"instance_id": "sf_bq010", "candidates": [invalid]}
        )


def test_official_sql_labels_exclude_ctes_and_resolve_two_part_names(
    tmp_path, catalog
):
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    task = {
        "instance_id": "sf_test",
        "db_id": "GA360",
        "instruction": "test",
    }
    # The production preflight evaluates every available official SQL file.
    tasks = {}
    for index in range(2):
        instance_id = f"sf_test_{index:03d}"
        tasks[instance_id] = {**task, "instance_id": instance_id}
        (sql_dir / f"{instance_id}.sql").write_text(
            """
            WITH selected AS (
              SELECT * FROM GOOGLE_ANALYTICS_SAMPLE.GA_SESSIONS_20170701
            )
            SELECT * FROM selected
            """,
            encoding="utf-8",
        )
    labels = extract_official_sql_labels(
        sql_dir=sql_dir,
        tasks=tasks,
        catalog=catalog,
    )
    assert len(labels) == 2
    assert labels[0]["physical_tables"] == [
        "GA360.GOOGLE_ANALYTICS_SAMPLE.GA_SESSIONS_20170701"
    ]


def test_task_set_is_complete_stable_and_hashed(catalog):
    labels = []
    for index in range(12):
        labels.append(
            {
                "instance_id": f"task_{index:02d}",
                "database_id": "GA360",
                "physical_tables": [
                    "GA360.GOOGLE_ANALYTICS_SAMPLE.GA_SESSIONS_20170701"
                ],
                "families": [
                    "GA360.GOOGLE_ANALYTICS_SAMPLE.GA_SESSIONS_{YYYYMMDD}"
                ],
                "sql_sha256": str(index),
            }
        )
    first = make_task_set(labels=list(reversed(labels)), catalog=catalog)
    second = make_task_set(labels=labels, catalog=catalog)
    assert first == second
    assert first["task_count"] == 12
    assert first["instance_ids"] == sorted(label["instance_id"] for label in labels)
    assert len(first["labels_sha256"]) == 64
    assert len(first["catalog_sha256"]) == 64


class _FakeCompletions:
    def __init__(self, submit_arguments, *, submit_on_call=6):
        self.calls = 0
        self.submit_arguments = submit_arguments
        self.submit_on_call = submit_on_call
        self.requests = []

    def create(self, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)
        if self.calls < self.submit_on_call:
            message = SimpleNamespace(content="Still exploring", tool_calls=[])
        else:
            call = SimpleNamespace(
                id="call_submit",
                function=SimpleNamespace(
                    name="submit_table_selection",
                    arguments=json.dumps(self.submit_arguments),
                ),
            )
            message = SimpleNamespace(content="", tool_calls=[call])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )


class _StatusError(RuntimeError):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _ErrorCompletions:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        message = SimpleNamespace(content="ok", tool_calls=[])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=1, completion_tokens=1, total_tokens=2
            ),
        )


class _ScriptedCompletions:
    def __init__(self, tool_arguments):
        self.tool_arguments = list(tool_arguments)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        arguments = self.tool_arguments.pop(0)
        calls = []
        if arguments is not None:
            calls = [
                SimpleNamespace(
                    id=f"call-{len(self.requests)}",
                    function=SimpleNamespace(
                        name="submit_table_selection",
                        arguments=json.dumps(arguments),
                    ),
                )
            ]
        message = SimpleNamespace(content="" if calls else "plain", tool_calls=calls)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(
                prompt_tokens=1, completion_tokens=1, total_tokens=2
            ),
        )


def _submit_arguments():
    return {
        "candidates": [
            {
                "rank": 1,
                "family_id": (
                    "GA360.GOOGLE_ANALYTICS_SAMPLE.GA_SESSIONS_{YYYYMMDD}"
                ),
                "tier": "required",
                "roles": ["fact"],
                "reason": "Needed",
                "variant_selector": {
                    "mode": "exact",
                    "tables": [
                        "GA360.GOOGLE_ANALYTICS_SAMPLE.GA_SESSIONS_20170701"
                    ],
                },
            }
        ]
    }


def _agent(completions, *, max_rounds=6, max_attempts=3):
    return SchemaRouterAgent(
        model_client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        ),
        model_config={
            "name": "mock",
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 1000,
            "retry": {
                "max_attempts": max_attempts,
                "initial_delay_seconds": 0,
                "backoff_multiplier": 1,
                "max_delay_seconds": 0,
            },
        },
        system_prompt="router",
        max_rounds=max_rounds,
        max_tool_calls=24,
    )


def _tools(catalog):
    return SchemaRouterTools(
        catalog,
        database_id="GA360",
        instance_id="sf_bq010",
        sample_rows=1,
        max_sample_chars=1000,
    )


def _run(agent, tools):
    return agent.run(
        item={
            "instance_id": "sf_bq010",
            "db_id": "GA360",
            "instruction": "July sessions",
        },
        rollout_idx=0,
        tools=tools,
        user_prompt="route",
    )


def test_last_round_forces_submission(catalog):
    completions = _FakeCompletions(_submit_arguments())
    result = _run(_agent(completions), _tools(catalog))

    assert result["completed"] is True
    assert result["instance_id"] == "sf_bq010"
    assert result["selection"]["instance_id"] == "sf_bq010"
    assert result["performance"]["forced_submissions"] == 1
    assert completions.calls == 6
    assert all(request["tool_choice"] == "auto" for request in completions.requests)
    final_tools = completions.requests[-1]["tools"]
    assert [tool["function"]["name"] for tool in final_tools] == [
        "submit_table_selection"
    ]
    assert set(final_tools[0]["function"]["parameters"]["properties"]) == {
        "candidates"
    }
    assert result["trace"][-1]["stage"] == "submission_only"


def test_router_prompt_does_not_ask_model_to_echo_instance_id():
    prompt = build_router_user_prompt(
        {
            "instance_id": "sf_bq010",
            "db_id": "GA360",
            "instruction": "July sessions",
        },
        external_knowledge=None,
        family_overview="families",
    )

    assert "sf_bq010" not in prompt


def test_plain_text_final_round_gets_one_submission_only_repair(catalog):
    completions = _FakeCompletions(_submit_arguments(), submit_on_call=7)
    result = _run(_agent(completions), _tools(catalog))
    assert result["completed"] is True
    assert result["performance"]["format_repairs"] == 1
    assert result["trace"][-1]["stage"] == "format_repair"
    for request in completions.requests[-2:]:
        assert request["tool_choice"] == "auto"
        assert [tool["function"]["name"] for tool in request["tools"]] == [
            "submit_table_selection"
        ]


def test_plain_text_repair_fails_without_full_table_fallback(catalog):
    completions = _FakeCompletions(_submit_arguments(), submit_on_call=99)
    result = _run(_agent(completions, max_rounds=2), _tools(catalog))

    assert result["completed"] is False
    assert "did not call submit_table_selection" in result["error"]
    assert result["trace"][-1]["stage"] == "format_repair"


def test_invalid_final_submission_can_be_repaired_once(catalog):
    completions = _ScriptedCompletions(
        [None, {"candidates": []}, _submit_arguments()]
    )
    result = _run(_agent(completions, max_rounds=2), _tools(catalog))

    assert result["completed"] is True
    assert result["performance"]["format_repairs"] == 1
    assert result["trace"][-1]["stage"] == "format_repair"
    assert result["selection"]["instance_id"] == "sf_bq010"


@pytest.mark.parametrize("status_code", [400, 422])
def test_non_retryable_model_errors_fail_immediately(status_code):
    completions = _ErrorCompletions([_StatusError(status_code)])
    agent = _agent(completions, max_attempts=3)

    with pytest.raises(_StatusError):
        agent._call_model([], tool_schemas=[])

    assert completions.calls == 1


@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 503])
def test_retryable_model_statuses_are_retried(status_code):
    completions = _ErrorCompletions([_StatusError(status_code)])
    agent = _agent(completions, max_attempts=3)

    _response, attempts = agent._call_model([], tool_schemas=[])

    assert attempts == 2
    assert completions.calls == 2


@pytest.mark.parametrize("error", [TimeoutError("timeout"), ConnectionError("down")])
def test_transport_errors_are_retried(error):
    completions = _ErrorCompletions([error])
    agent = _agent(completions, max_attempts=3)

    _response, attempts = agent._call_model([], tool_schemas=[])

    assert attempts == 2


def test_scores_keep_rollouts_separate(catalog):
    family_id = "GA360.GOOGLE_ANALYTICS_SAMPLE.GA_SESSIONS_{YYYYMMDD}"
    table = "GA360.GOOGLE_ANALYTICS_SAMPLE.GA_SESSIONS_20170701"
    label = {
        "instance_id": "sf_bq010",
        "database_id": "GA360",
        "physical_tables": [table],
        "families": [family_id],
    }
    complete = {
        "instance_id": "sf_bq010",
        "rollout_idx": 0,
        "completed": True,
        "selection": {
            "candidates": [
                {
                    "family_id": family_id,
                    "tier": "required",
                    "resolved_physical_tables": [table],
                }
            ],
            "resolved_physical_tables": [table],
        },
        "performance": {},
    }
    failed = {
        "instance_id": "sf_bq010",
        "rollout_idx": 1,
        "completed": False,
        "error": "failed",
        "performance": {},
    }
    scores = [
        score_rollout(result=complete, label=label, catalog=catalog),
        score_rollout(result=failed, label=label, catalog=catalog),
    ]
    summary = aggregate_scores(
        scores=scores, expected_instance_ids=["sf_bq010"]
    )
    assert summary["physical_task_full_coverage"] == 0.5
    assert summary["stability"]["instances_with_all_rollouts_physical_full"] == 0


def test_full_evaluation_threshold_and_report():
    summary = {
        "official_sql_files": 120,
        "total_tasks": 120,
        "rollouts_per_task": 1,
        "expected_rollouts": 120,
        "scored_rollouts": 120,
        "completed_rollouts": 119,
        "failed_rollouts": 1,
        "physical_task_full_coverage": 0.98,
        "physical_micro_recall": 0.995,
        "physical_macro_recall": 0.99,
        "family_task_full_coverage": 0.99,
        "family_micro_recall": 0.999,
        "family_macro_recall": 0.999,
        "average_physical_compression": 0.9,
        "average_family_compression": 0.8,
        "average_candidate_rendered_chars": 100.0,
        "invalid_references": 0,
        "family_recall_at_k": {str(k): 1.0 for k in (1, 5, 10, 20)},
        "physical_recall_at_k": {str(k): 1.0 for k in (1, 5, 10, 20)},
        "tier_physical_recall": {
            "required": 1.0,
            "supporting": 0.0,
            "possible": 0.0,
        },
        "stability": {
            "instances_with_all_rollouts_physical_full": 118,
            "instances": 120,
            "worst_rollout_physical_recall": 0.0,
        },
        "performance": {
            "model_calls": 120,
            "tool_calls": 300,
            "total_tokens": 1000,
            "duration_seconds": 10.0,
        },
    }
    threshold = threshold_status(
        summary=summary,
        thresholds={
            "physical_task_full_coverage": 0.97,
            "physical_micro_recall": 0.99,
            "invalid_references": 0,
        },
    )
    report = render_report(summary=summary, threshold=threshold)

    assert threshold["passed"] is True
    assert "全量评测报告" in report
    assert "预期 Rollout：120" in report
    assert "失败 Rollout：1" in report
    assert "development" not in report
    assert "holdout" not in report
    assert "留出集" not in report
