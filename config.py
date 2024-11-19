import yaml
def load_config():
    with open('./config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg
    