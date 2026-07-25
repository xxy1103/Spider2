# Spider 2.0-Snow

为了与研究中对“传统 Text2SQL 设置”的兴趣保持一致，并让评估更加方便，我们已将 Spider 2.0 中所有非 DBT 项目所使用的数据库托管到了 Snowflake（感谢 Snowflake 的支持）。在这种设置下，用户只需使用一种 SQL 方言即可完成任务，从而让文本到 SQL 的研究更加聚焦。

## 🚀 快速开始

1. **Snowflake 账户**：按照这个[指南](https://github.com/xlang-ai/Spider2/blob/main/assets/Snowflake_Guideline.md)为你的 Snowflake 数据库获取自己的用户名和密码。你需要更新 `bigquery_credential.json` 和 `snowflake_credential.json`。
2. 更新 `bigquery_credential.json` 和 `snowflake_credential.json`。

### 运行 Spider-Agent（Snow）

1. **安装 Docker**。按照 [Docker 安装指南](https://docs.docker.com/engine/install/) 在你的机器上安装 Docker。
2. **安装 conda 环境**。

```bash
git clone https://github.com/xlang-ai/Spider2.git
cd methods/spider-agent-snow

# 可选：为 Spider 2.0 创建一个 Conda 环境
# conda create -n spider2 python=3.11
# conda activate spider2

# 安装所需依赖
pip install -r requirements.txt
```

3. **配置凭据**：按照这个[指南](https://github.com/xlang-ai/Spider2/blob/main/assets/Snowflake_Guideline.md)获取你在 Snowflake 数据库中的用户名和密码。你需要更新 `snowflake_credential.json`。
4. **Spider 2.0-Snow 配置**

```bash
python spider_agent_setup_snow.py
```

5. **运行 Agent**

```bash
export OPENAI_API_KEY=your_openai_api_key
python run.py --model gpt-4o -s test1
```

### 在 Spider2-Snow 上运行 DAIL-SQL

1. 将 `spider2-lite/baselines/dailsql` 文件夹复制到 `spider-snow/baselines/dailsql`。
2. 将 `spider-snow/baselines/dailsql/run.sh` 中的 `DEV=spider2-lite` 改为 `DEV=spider2-snow`。
3. 以与 `spider-lite` 相同的方式在 `spider2-snow` 上运行 DAIL-SQL。

## 评估

```bash
python get_spider2snow_submission_data.py --experiment_suffix gpt-4o-test1 --results_folder_name ../../spider2-snow/evaluation_suite/gpt-4o-test1

cd ../../spider2-snow/evaluation_suite
python evaluate.py --mode exec_result --result_dir gpt-4o-test1
```
