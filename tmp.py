# import nltk
# nltk.download('punkt')

import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

min_delta_val = cfg.get("early_stopping", {}).get("min_delta", 1e-4)
print(min_delta_val, type(min_delta_val))