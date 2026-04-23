import os
import json

def load_feedback_history(dataset_hash, cache_dir="model_cache"):
    filename = f"feedback_{dataset_hash}.json"
    path = os.path.join(cache_dir, filename)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                if not isinstance(data, dict) or "residuals" not in data or "maes" not in data:
                    raise ValueError("Invalid format")
                return data
        except Exception:
            return {"residuals": [], "maes": []}
    return {"residuals": [], "maes": []}

def save_feedback_history(data, dataset_hash, cache_dir="model_cache"):
    filename = f"feedback_{dataset_hash}.json"
    path = os.path.join(cache_dir, filename)
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data, f)
    os.replace(tmp_path, path)
