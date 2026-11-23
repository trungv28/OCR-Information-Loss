import yaml

with open('config/params.yaml', encoding='utf-8') as f:
    PARAMS = yaml.safe_load(f)
with open('config/prompts.yaml', encoding='utf-8') as f:
    PROMPTS = yaml.safe_load(f)
