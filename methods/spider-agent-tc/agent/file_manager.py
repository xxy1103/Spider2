import json
import logging
import os
import threading
import glob
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

class FileManager:
    def __init__(self, args):
        self.args = args
        self.file_locks = defaultdict(threading.Lock)
        self.processed_instances = defaultdict(int)
        
    def check_if_terminated(self, result):
        """Check if a result has successfully executed to termination"""
        return result.get("terminated", False)
        
    def get_instance_file_path(self, instance_id):
        """Get the file path for a specific instance"""
        return os.path.join(self.args.output_folder, f"{instance_id}.json")
        
    def load_instance_results(self, instance_id):
        """Load results for a specific instance"""
        file_path = self.get_instance_file_path(instance_id)
        if not os.path.exists(file_path):
            return []
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        except Exception as e:
            logger.exception("Error loading %s: %s", file_path, e)
            return []
    
    def save_instance_results(self, instance_id, results):
        """Atomically save all results for a specific instance."""
        file_path = Path(self.get_instance_file_path(instance_id))
        os.makedirs(self.args.output_folder, exist_ok=True)
        temporary_path = file_path.with_name(
            f".{file_path.name}.{threading.get_ident()}.tmp"
        )

        try:
            with open(temporary_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_path, file_path)
        except Exception as e:
            logger.exception("Error saving %s: %s", file_path, e)
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove temporary result file %s", temporary_path)

    def upsert_rollout_result(self, result):
        """Insert or refresh one rollout without duplicating progress snapshots."""
        instance_id = result["instance_id"]
        rollout_idx = result["rollout_idx"]

        with self.file_locks[instance_id]:
            existing_results = self.load_instance_results(instance_id)
            previous_result = None
            for index, existing in enumerate(existing_results):
                if (
                    isinstance(existing, dict)
                    and existing.get("rollout_idx") == rollout_idx
                ):
                    previous_result = existing
                    existing_results[index] = result
                    break
            else:
                existing_results.append(result)

            self.save_instance_results(instance_id, existing_results)

            was_terminated = bool(
                previous_result and self.check_if_terminated(previous_result)
            )
            if self.check_if_terminated(result) and not was_terminated:
                self.processed_instances[instance_id] += 1
    
    def add_single_result(self, result):
        """Add a single result to the appropriate instance file"""
        instance_id = result["instance_id"]
        
        with self.file_locks[instance_id]:
            existing_results = self.load_instance_results(instance_id)
            existing_results.append(result)
            self.save_instance_results(instance_id, existing_results)
            
            if self.check_if_terminated(result):
                self.processed_instances[instance_id] += 1
        
    def load_existing_results(self):
        """Load existing results from all instance files"""
        if not os.path.exists(self.args.output_folder):
            return []
            
        all_results = []
        instance_files = glob.glob(os.path.join(self.args.output_folder, "*.json"))
        infrastructure_files = {
            "run-manifest.json",
            "run-summary.json",
            "selected-tasks.json",
            "failed-tasks.json",
            "routing-index.json",
        }
        instance_files = [
            path
            for path in instance_files
            if os.path.basename(path) not in infrastructure_files
        ]
        
        for file_path in instance_files:
            try:
                filename = os.path.basename(file_path)
                instance_id = filename.replace('.json', '')
                
                instance_results = self.load_instance_results(instance_id)
                terminated_count = 0
                
                for result in instance_results:
                    if isinstance(result, dict) and "instance_id" in result:
                        all_results.append(result)
                        if self.check_if_terminated(result):
                            terminated_count += 1
                
                self.processed_instances[instance_id] = terminated_count
                        
            except Exception as e:
                logger.exception("Error processing %s: %s", file_path, e)
                continue

        return all_results
