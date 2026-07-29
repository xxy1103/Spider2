# Tool-call Agent for Spider 2.0-Snow

This implementation runs Spider Agent TC from one YAML configuration. The launcher
selects tasks, validates Snowflake and model access, starts the local tool server,
runs the agent, writes reproducibility records, and always stops the tool server.

The supported entrypoint is:

```bash
python run.py --config configs/smoke.yaml
```

`run.sh` and the former collection of command-line overrides have been removed.

## WSL and Conda setup

Run the project from WSL because the schema exploration tool expects Linux shell
commands.

```bash
cd /mnt/c/Users/ulna/Desktop/Spider2/methods/spider-agent-tc
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
your OpenAI-compatible service. The default task selection contains only
`sf_bq011`, with one thread and one rollout.

```bash
conda activate spider2-tc
cd /mnt/c/Users/ulna/Desktop/Spider2/methods/spider-agent-tc
python run.py --config configs/smoke.yaml
```

The launcher performs all static checks first, then verifies Snowflake, performs
a minimal model request, starts the tool server on port 5000 or the next available
local port, and runs the selected task.

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
- `run-summary.json`: completion summary;
- `failed-tasks.json`: IDs that did not complete every requested rollout.
- `run.log`: detailed Agent, tool-server, and evaluation diagnostics. The
  terminal stays focused on aggregate progress and writes credential-redacted
  details here.

The process exits with status 1 when any selected task remains incomplete, status
2 for configuration or startup failures, and status 130 when interrupted.

## Run without a Snowflake account

Mock mode exercises configuration loading, task selection, the real model,
tool-server lifecycle, XML tool calls, conversation persistence, and result
summaries without connecting to Snowflake.

Create a model-only secrets file:

```bash
cp configs/secrets.mock.example.yaml configs/secrets.yaml
```

Set your real model endpoint and API key in `configs/secrets.yaml`, then set the
real model ID in `configs/mock.yaml` and run:

```bash
python run.py --config configs/mock.yaml
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
  snowflake:
    mode: mock
    mock:
      response_csv: |
        COLUMN_NAME
        fake-value
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
- `tools`: Bash settings plus live/mock Snowflake mode, execution timeout, and returned-output limits.
- `preflight`: enable or disable live model and Snowflake connectivity checks.

Unknown fields and incorrect types are rejected to prevent silently ignored
configuration mistakes.

## Export submission SQL

The submission converter is unchanged:

```bash
python convert_to_submission_format.py \
  results/smoke-test \
  ../../spider2-snow/evaluation_suite/smoke-test
```
