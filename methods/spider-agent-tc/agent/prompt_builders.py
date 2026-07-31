import os

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
    def build_initial_prompt(self, item, args):
        system_prompt = self.load_system_prompt(args)
        external_knowledge_content = self.load_external_knowledge(item.get('external_knowledge'), args)
        
        user_content = f"""Question: {item['instruction']}
External Knowledge: {external_knowledge_content if external_knowledge_content else 'None'}

The allowed database for this task is {item['db_id']}. Use the structured tools
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
