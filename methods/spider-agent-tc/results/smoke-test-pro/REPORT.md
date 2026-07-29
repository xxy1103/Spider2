# Agent 运行报告

**实验名称**: `smoke-test-pro`  
**运行时间**: 2026-07-29T10:32:42.256086+00:00  
**模型**: `deepseek-v4-pro` (temperature=0, max_tokens=100000)  
**Agent 配置**: max_rounds=20, num_threads=1, rollout_number=1  

---

## 📊 运行概览

| 指标 | 数值 |
|------|------|
| 任务总数 | 1 |
| 成功任务 | 1 ✅ |
| 失败任务 | 0 ❌ |
| 成功率 | 100.00% |

---

## 🎯 评分结果

| 指标 | 数值 |
|------|------|
| 正确任务 | 1 / 1 |
| 本次准确率 | 100.00% |
| 全数据集得分率 | 0.18% (1/547) |

### 任务得分详情

| Instance ID | 得分 | 对话轮次 | 工具调用 | 状态 | 错误信息 |
|-------------|------|----------|----------|------|----------|
| sf_bq011 | ✅ 1 | 11 | 11 | 成功 | - |

---

## 🔧 工具使用统计

| 工具名称 | 调用次数 | 占比 |
|----------|----------|------|
| execute_snowflake_sql | 6 | 54.5% |
| execute_bash | 4 | 36.4% |
| terminate | 1 | 9.1% |

---

## 📈 对话统计

| 指标 | 平均值 |
|------|--------|
| 对话轮次 | 11.0 |
| 工具调用次数 | 11.0 |
| 消息总数 | 23.0 |

---

## ❌ 失败分析

无失败任务。🎉

---

## 📝 配置文件

```yaml
experiment:
  name: smoke-test-pro
  results_root: methods/spider-agent-tc/results
  resume: true
secrets_file: methods/spider-agent-tc/configs/secrets.yaml
paths:
  input_file: spider2-snow/spider2-snow.jsonl
  databases: spider2-snow/resource/databases
  documents: spider2-snow/resource/documents
  system_prompt: methods/spider-agent-tc/prompts/spider_agent.txt
tasks:
  instance_ids:
  - sf_bq011
  index_ranges: []
  databases: []
  sample_size: null
  seed: 42
  order: seeded_shuffle
model:
  name: deepseek-v4-pro
  temperature: 0
  top_p: 1
  max_tokens: 100000
  request_timeout_seconds: 120
  retry:
    max_attempts: 3
    initial_delay_seconds: 1
    backoff_multiplier: 2
    max_delay_seconds: 10
agent:
  max_rounds: 20
  num_threads: 1
  rollout_number: 1
server:
  host: 127.0.0.1
  preferred_port: 5000
  workers_per_tool: 2
  startup_timeout_seconds: 15
  request_timeout_seconds: 90
tools:
  bash:
    timeout_seconds: 30
    max_output_chars: 2000
  snowflake:
    mode: live
    timeout_seconds: 60
    max_output_chars: 2000
preflight:
  check_model: true
  check_snowflake: true
auto_evaluate:
  enabled: true
  timeout: 300
  max_workers: 4

```

---

**报告生成时间**: 2026-07-29T18:35:38.274872  
**结果目录**: `/mnt/c/Users/ulna/Desktop/Spider2/methods/spider-agent-tc/results/smoke-test-pro`  