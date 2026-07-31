from collections import defaultdict
import os
import sqlite3
from pathlib import Path

class BasePromptBuilder:
    
    def load_system_prompt(self, args):
        """Load system prompt from file"""
        with open(args.system_prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def load_external_knowledge(self, external_knowledge_file, args):
        """Load external knowledge from file"""
        if not external_knowledge_file:
            return None
        
        knowledge_path = os.path.join(args.documents_path, external_knowledge_file)
        if os.path.exists(knowledge_path):
            with open(knowledge_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        return None
    
    
    def build_initial_prompt(self, item, args):
        raise NotImplementedError

class SpiderAgentPromptBuilder(BasePromptBuilder):
    @staticmethod
    def load_schema_overview(item, args):
        catalog_path = (
            Path(args.output_folder) / "tool-state" / "schema-index.sqlite"
        )
        if not catalog_path.is_file():
            raise RuntimeError(
                f"Schema catalog is missing for this run: {catalog_path}"
            )

        uri = catalog_path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                """
                SELECT schema_name, table_name
                FROM tables
                WHERE database_id = ?
                ORDER BY schema_name, table_name
                """,
                (item["db_id"],),
            ).fetchall()
        finally:
            connection.close()

        if not rows:
            raise RuntimeError(
                f"No indexed schemas found for database: {item['db_id']}"
            )
        tables_by_schema = defaultdict(list)
        for row in rows:
            tables_by_schema[row["schema_name"]].append(row["table_name"])

        lines = []
        for schema_name, table_names in tables_by_schema.items():
            lines.append(f"- {schema_name} ({len(table_names)} tables):")
            lines.extend(f"  - {table_name}" for table_name in table_names)
        return "\n".join(lines)

    def build_initial_prompt(self, item, args):
        system_prompt = self.load_system_prompt(args)
        external_knowledge_content = self.load_external_knowledge(item.get('external_knowledge'), args)
        schema_overview = self.load_schema_overview(item, args)
        
        user_content = f"""Question: {item['instruction']}
External Knowledge: {external_knowledge_content if external_knowledge_content else 'None'}

The allowed database for this task is {item['db_id']}.
Indexed schema overview:
{schema_overview}

Use the structured tools
to inspect only this task's schema and data. Every physical table reference must
use database_name.schema_name.table_name. Submit only a complete SQL query that
exactly matches a successful execution."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]



def get_prompt_builder(strategy):
    builders = {
        "spider-agent": SpiderAgentPromptBuilder(),
        # "database": DatabasePromptBuilder(),
        # "multi_step": MultiStepPromptBuilder(),
        # "reasoning": ReasoningPromptBuilder(),
    }
    
    return builders.get(strategy, SpiderAgentPromptBuilder())
