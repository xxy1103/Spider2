import snowflake.connector
from snowflake.connector.errors import ProgrammingError, DatabaseError
import pandas as pd
from typing import Dict, Any, Tuple
import logging
import time

logger = logging.getLogger(__name__)

TIMEOUT = 60
MAX_CSV_CHARS = 2000
SNOWFLAKE_CREDENTIALS = None
SNOWFLAKE_MODE = "live"
MOCK_RESPONSE_CSV = ""


def configure(config):
    global TIMEOUT, MAX_CSV_CHARS, SNOWFLAKE_CREDENTIALS, SNOWFLAKE_MODE, MOCK_RESPONSE_CSV
    settings = config.raw["tools"]["snowflake"]
    TIMEOUT = settings["timeout_seconds"]
    MAX_CSV_CHARS = settings["max_output_chars"]
    SNOWFLAKE_MODE = settings["mode"]
    MOCK_RESPONSE_CSV = settings.get("mock", {}).get("response_csv", "")
    SNOWFLAKE_CREDENTIALS = (
        dict(config.secrets["snowflake"]) if SNOWFLAKE_MODE == "live" else None
    )


def _redact(message: str) -> str:
    if not SNOWFLAKE_CREDENTIALS:
        return message
    for key in ("user", "password"):
        value = SNOWFLAKE_CREDENTIALS.get(key)
        if value:
            message = message.replace(value, "***REDACTED***")
    return message

def execute_snowflake_sql(sql: str, **kwargs) -> Dict[str, Any]:
    if SNOWFLAKE_MODE == "mock":
        return {
            "content": (
                "EXECUTION RESULT of [execute_snowflake_sql]:\n"
                "MOCK MODE: no Snowflake query was executed. "
                "This result must not be used for evaluation.\n\n"
                f"```csv\n{MOCK_RESPONSE_CSV}\n```"
            )
        }

    timeout = kwargs.get('timeout', TIMEOUT)
    start_time = time.time()
    
    content = ""
    
    conn = None
    try:
        # Get Snowflake credentials from file
        if SNOWFLAKE_CREDENTIALS is None:
            raise RuntimeError("Snowflake tool was not configured")
        
        # Connect to Snowflake using credentials
        conn = snowflake.connector.connect(
            **SNOWFLAKE_CREDENTIALS,
            login_timeout=timeout,
            network_timeout=timeout
        )
        cursor = conn.cursor()
        
        # Execute SQL query
        cursor.execute(sql)
        
        # First print success message
        print("Query executed successfully")
        
        # Fetch results if the query returns data
        if cursor.description:
            headers = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            if rows:
                df = pd.DataFrame(rows, columns=headers)
                
                # Convert full dataset to CSV
                full_csv_data = df.to_csv(index=False)
                total_rows = len(df)
                
                # Check if we need to truncate by character length
                if len(full_csv_data) > MAX_CSV_CHARS:
                    # Truncate to MAX_CSV_CHARS characters
                    truncated_csv = full_csv_data[:MAX_CSV_CHARS]
                    
                    # Find the last complete line to avoid cutting in the middle
                    last_newline = truncated_csv.rfind('\n')
                    if last_newline > 0:
                        truncated_csv = truncated_csv[:last_newline]
                    
                    content = f"""Query executed successfully

```csv
{truncated_csv}
```

Note: The result has been truncated to {MAX_CSV_CHARS} characters for display purposes. The complete result set contains {total_rows} rows and {len(full_csv_data)} characters."""
                else:
                    content = f"""Query executed successfully

```csv
{full_csv_data}
```"""
            else:
                content = "Query executed successfully, but no rows returned."
        else:
            conn.commit()
            content = "Query executed successfully."
        
        
    except ProgrammingError as e:
        content = f"SQL Error: {_redact(str(e))}"
        logger.error("Snowflake SQL programming error")
    except DatabaseError as e:
        content = f"Database error: {_redact(str(e))}"
        logger.error("Snowflake database error")
    except TimeoutError:
        content = f"Execution timed out after {timeout} seconds."
        logger.error(f"Snowflake query timed out: {sql}")
    except Exception as e:
        content = f"Unexpected error: {_redact(str(e))}"
        logger.error(f"Unexpected Snowflake execution error: {type(e).__name__}")
    finally:
        if conn:
            conn.close()
            
        # Log execution time
        execution_time = time.time() - start_time
        logger.info(f"Execution completed in {execution_time:.2f} seconds")
    
    return {
        "content": f"EXECUTION RESULT of [execute_snowflake_sql]:\n{content}"
    }

def register_tools(registry):
    registry.register_tool("execute_snowflake_sql", execute_snowflake_sql)
