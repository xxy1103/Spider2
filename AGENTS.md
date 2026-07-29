# Repository Guidelines

## 项目结构与修改范围

仓库包含三套基准数据与评测目录：`spider2-snow/`、`spider2-lite/` 和 `spider2-dbt/`；`assets/` 存放说明文档与图片；`methods/` 收录各类 Agent 实现。本分支默认只维护 `methods/spider-agent-tc/`，其源码、测试、提示词、文档和实验 YAML 可按任务修改。其余目录及仓库顶层文件默认只读，除非用户明确点名。保留工作区中已有的未提交改动，尤其不要覆盖实验配置。

## 环境、运行与检查

Codex 默认在 Windows PowerShell 中工作。进入 `methods/spider-agent-tc` 后，使用 Conda 环境完成本地开发与离线验证：

```powershell
conda env create -f environment.yml
conda activate spider2-tc
python run.py --config configs/smoke.yaml
python -m pytest tests -q
python -m compileall .
git diff --check
```

启动方式必须保持“单一 YAML 配置”：入口只接受 `--config`。不要恢复 `run.sh`，不要增加零散 CLI 覆盖项，也不要引入无限重试。联网模型调用、Snowflake 查询及可能产生费用的 Live 运行必须先获得用户明确授权。

## 代码风格与测试

Python 使用 4 空格缩进；函数、变量和测试使用清晰的 `snake_case`，类使用 `PascalCase`。遵循相邻代码的导入、类型标注和异常处理风格。测试使用 `pytest`，放在 `methods/spider-agent-tc/tests/`，文件命名为 `test_*.py`。行为变更应覆盖正常路径、配置校验和失败路径。

Mock 运行只验证配置、模型调用、工具生命周期和结果持久化，不代表 SQL 正确性或基准成绩。不得将 Mock 输出用于排行榜或性能比较。检查失败或无法执行时，应准确说明错误和未验证项，不得宣称通过。

## 安全、结果与 Git

禁止读取、修改或输出 `methods/spider-agent-tc/configs/secrets.yaml`。`results/` 中的脱敏产物仅可用于检查，不得手工改写。提交信息沿用 `feat:`、`fix:` 等前缀，可使用简洁中文主题。只有用户明确要求时才创建本地 commit；不得自动 push 或创建 PR，也不得在验证失败时提交。

PR 应说明改动目的、使用的实验配置、验证证据以及运行属于 Mock 还是 Live；行为或输出变化需附简短结果摘要。
