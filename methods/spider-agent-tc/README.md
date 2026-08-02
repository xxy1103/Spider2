# Tool-call Agent for Spider 2.0-Snow

This implementation runs Spider Agent TC from one YAML configuration. The launcher
selects tasks, validates Snowflake and model access, starts the local tool server,
runs the agent, writes reproducibility records, and always stops the tool server.

The supported entrypoint is:

```bash
python run.py --config configs/smoke.yaml
```

`run.sh` and the former collection of command-line overrides have been removed.

## Conda setup

The Agent no longer receives a general-purpose shell tool. The structured tool
server works from Windows PowerShell or WSL.

```bash
cd methods/spider-agent-tc
conda env create -f environment.yml
conda activate spider2-tc
```

To update an existing environment:

```bash
conda env update -f environment.yml --prune
```

## Secrets

Create the local secrets file:

```bash
cp configs/secrets.example.yaml configs/secrets.yaml
```

Edit `configs/secrets.yaml`:

```yaml
model_api:
  base_url: https://your-openai-compatible-service.example/v1
  api_key: your-api-key

snowflake:
  user: your-spider2-user
  password: your-generated-python-token
  account: RSRSBDK-YDB67606
  role: PARTICIPANT
  warehouse: COMPUTE_WH_PARTICIPANT
```

Use the Python token described in the repository Snowflake guideline, not
necessarily the password used in the web UI. The real `secrets.yaml` is ignored by
Git. The launcher redacts the model key, Snowflake user, and Snowflake token from
experiment snapshots and error messages.

## First run

Edit `model.name` in `configs/smoke.yaml` so it matches the model ID exposed by
your OpenAI-compatible service. The checked-in smoke configuration selects task
indexes 0 through 49, with one thread and one rollout per task.

```bash
conda activate spider2-tc
cd methods/spider-agent-tc
python run.py --config configs/smoke.yaml
```

The launcher performs all static checks first, rebuilds a temporary SQLite schema
index for only the databases referenced by the selected tasks, verifies
Snowflake, performs a minimal model request, starts the tool server on port 5000
or the next available local port, and runs the selected tasks.

Results are written to:

```text
results/smoke-test/YYYYMMDD-HHMMSS/
```

`experiment.name` identifies an experiment group. Every completed invocation
gets its own timestamped run directory, which contains per-task conversation
JSON files plus:

- `effective-config.yaml`: resolved configuration with secrets redacted;
- `selected-tasks.json`: exact task order;
- `run-manifest.json`: configuration fingerprint and environment versions;
- `run-summary.json`: completion summary plus aggregate and per-task performance
  profiles;
- `failed-tasks.json`: IDs that did not complete every requested rollout.
- `run.log`: detailed Agent, tool-server, and evaluation diagnostics. The
  terminal stays focused on aggregate progress and writes credential-redacted
  details here.
- `tool-state/schema-index.sqlite`: run-local searchable metadata for only the
  selected databases.
- `tool-state/<instance>/<rollout>/`: task-scoped SQL artifacts and execution
  records. These files support validation and are not used directly by scoring.

Each per-task conversation JSON file is refreshed atomically after every
completed Agent graph step. Its current rollout record has
`"in_progress": true` while the task is running and changes to `false` after
the rollout finishes, so the file can be watched safely during execution.

The process exits with status 1 when any selected task remains incomplete, status
2 for configuration or startup failures, and status 130 when interrupted.

Completed per-task records contain a `performance` object with wall-clock task
duration, cumulative model and tool duration, model retry/error counts, SQL
call/error counts, maximum executed or submitted SQL length, and rejected
`terminate` calls. `run-summary.json` aggregates these fields and records Agent
wall-clock time, average/P95 task duration, the slowest task, and tool-level
duration totals. When automatic evaluation is enabled, `REPORT.md` includes the
same overview and a table of the ten slowest tasks. Tool durations are measured
inside the tool server; cumulative durations can therefore exceed Agent
wall-clock time when calls or tasks run concurrently.

## Run without a Snowflake account

Mock mode exercises configuration loading, task selection, the real model,
structured tool calls, tool-server lifecycle, conversation persistence, and
result summaries without connecting to Snowflake. Copy `configs/smoke.yaml` to
an untracked experiment YAML, set `tools.sql.mode: mock`, and provide only the
model credentials required by your endpoint:

```bash
python run.py --config path/to/your-mock.yaml
```

Mock mode returns this deterministic fake query result:

```csv
MOCK_RESULT
1
```

No SQL is executed and the generated SQL is not evidence of correctness. Tool
output and `run-summary.json` explicitly identify the run as Mock. Never submit or
compare Mock results as benchmark results.

To customize the fake response, edit:

```yaml
tools:
  sql:
    mode: mock
    mock:
      response_csv: |
        COLUMN_NAME
        fake-value
preflight:
  check_model: true
  check_snowflake: false
auto_evaluate:
  enabled: false
  timeout: 300
  max_workers: 4
```

For a real run, use `configs/smoke.yaml`, whose Snowflake mode is `live`, and
provide the complete Snowflake section in `configs/secrets.yaml`.

## Selecting tasks

Task filters are applied as an intersection in this order:

1. `instance_ids`;
2. zero-based `index_ranges`;
3. `databases`;
4. deterministic sampling;
5. deterministic shuffle.

Examples:

```yaml
tasks:
  instance_ids: []
  index_ranges:
    - "0-9"
    - "20"
  databases:
    - GA4
  sample_size: 5
  seed: 42
  order: seeded_shuffle
```

An empty list means that filter is not applied. Leave all filters empty and set
`sample_size: null` to run all 547 tasks:

```yaml
tasks:
  instance_ids: []
  index_ranges: []
  databases: []
  sample_size: null
  seed: 42
  order: seeded_shuffle
```

The selector rejects unknown IDs or databases, invalid ranges, empty
intersections, and samples larger than the filtered candidate set before making a
model request.

## Resume behavior

With `experiment.resume: true`, the launcher resumes the newest timestamped run
that has the same configuration fingerprint and no `run-summary.json`. Tasks
with the requested number of terminated rollouts are skipped, while failed and
incomplete tasks run again. Once a run writes its summary, the next invocation
creates a new timestamped directory. With `resume: false`, every invocation
creates a new timestamped directory.

Older flat result directories remain readable and are not moved automatically.

## Configuration reference

All repository paths are resolved relative to the repository root, regardless of
the current working directory.

- `experiment`: result directory name and resume policy.
- `secrets_file`: repository-relative path to the untracked secrets YAML.
- `paths`: input JSONL, schemas, documents, and system prompt.
- `tasks`: filters, sample size, seed, and ordering.
- `model`: model ID, sampling, request timeout, and finite retry policy.
- `agent`: rounds, task concurrency, and rollout count.
- `server`: bind address, preferred port, workers, startup timeout, and request timeout.
- `tools.catalog`: schema-search page size plus local sample row and character
  limits. Oversized values are returned as bounded previews with truncation
  metadata.
- `tools.sql`: live/mock Snowflake mode, timeout, preview/pagination limits, and
  maximum SQL length.
- `tools.submission`: exact-execution and non-empty-result submission policy.
- `preflight`: enable or disable live model and Snowflake connectivity checks.

Unknown fields and incorrect types are rejected to prevent silently ignored
configuration mistakes. Legacy `tools.bash` and `tools.snowflake` blocks fail
with an explicit migration error.

## Structured tool workflow

The model can use only these task-scoped tools:

The initial user message contains the question, external knowledge, allowed
database, and the complete indexed table-name list grouped by schema.

1. `search_schema` and `describe_table` to find and inspect relevant tables.
2. `resolve_table_set` and `build_union_sql` to construct complete deterministic
   SQL for date-sharded tables without abbreviations.
3. `execute_sql` to validate scope and run one read-only Snowflake query, returning
   at most 20 preview rows by default.
4. `read_query_result` for explicit bounded pagination and `get_sql_text` to
   recover an untruncated SQL artifact.
5. `terminate(answer="<complete SQL>")` to submit the exact SQL text.

Every call is bound to an instance, rollout, and allowed `db_id`. Cross-database
references are rejected before Snowflake is contacted. A live submission is
accepted only when its exact byte content has already executed successfully and
returned at least one row. Mock executions can never satisfy this submission
gate. Reaching `agent.max_rounds` without an accepted `terminate` leaves the
rollout incomplete. These checks prevent incomplete or unexecuted submissions;
they do not resolve semantic misunderstandings of the question, which remains
the Agent's responsibility.

## 主 Agent 的 Schema Router 硬隔离

`run.py` 会先为每个“题目 × rollout”运行 Schema Router，再启动主 Agent。
Router 可读取完整本地 Schema；主 Agent 的初始 Schema 清单只包含 Router
提交的 `required`、`supporting`、`possible` 候选展开后的物理表并集。

同一物理表白名单同时约束 `search_schema`、`describe_table`、
`resolve_table_set`、SQL 构建、执行和 `terminate`。Router 失败、空提交或非法
selection 的 rollout 直接失败，不回退完整数据库。题目和 external knowledge
保持原文，但其中出现的白名单外表名也不能通过工具或 SQL 访问。

主运行目录中的 `routing-index.json` 保存逐 rollout 白名单和压缩前后表数；
`routing/<instance_id>/rollout-<n>.json` 保存 Router 审计轨迹。配置仍通过主
YAML 的 `schema_router` 段统一管理，入口仍然只接受 `--config`。

## 独立 Schema Router 评测

`run_schema_router.py` 在主 Agent 之外运行，只读取本地 Schema、DDL、样例行
和 external knowledge，不连接 Snowflake，也不执行 SQL。它用仓库公开的 120
份官方 SQL 生成评测标签；官方 SQL 和标签不会进入 Router 的模型上下文。

```powershell
python run_schema_router.py --config configs/schema-router.yaml
```

通过 `conda run` 启动时可加 `--no-capture-output`，使 Rich 聚合进度即时显示。
进度只展示已结束、进行中、成功、失败、续跑跳过、已运行时间和题/小时；详细
模型及工具错误写入 `run.log`。

配置入口仍然只接受 `--config`。每次运行当前全部可解析的官方 SQL；当前为
120 题。每题运行次数由 `evaluation.rollouts` 控制，默认 1。正式模型调用前
仍需明确授权。

Router 输出位于同一实验目录：

- `schema-router-labels.jsonl`：全部官方 SQL 表标签；
- `schema-router-task-set.json`：全量任务 ID 及 catalog/label 哈希；
- `routing/<instance_id>/rollout-<n>.json`：候选表族、物理表展开和审计轨迹；
- `schema-router-summary.json` 与 `schema-router-report.md`：召回、压缩、稳定性
  和性能汇总。

Router 只拥有元数据工具：表族列表、候选搜索、表族描述、变体解析和最终提交。
它没有 SQL 执行工具。独立入口只生成专项评测；主入口会消费相同 selection
协议并对主 Agent 实施硬白名单。
最终提交中的 `instance_id` 由运行器绑定，模型只提交候选表族。最后一轮及唯一
一次格式修复只暴露最终提交工具，以兼容不支持指定 `tool_choice` 的 reasoning
接口。运行指纹包含 Router 协议版本、Prompt 内容、全量任务清单和 Catalog
哈希，行为或评测数据变化不会续跑进旧版本目录。全量报告统一应用覆盖率与
非法引用门槛，未达标时报告仍会落盘，命令返回状态码 2。

## Export submission SQL

The submission converter is unchanged:

```bash
python convert_to_submission_format.py \
  results/smoke-test \
  ../../spider2-snow/evaluation_suite/smoke-test
```
