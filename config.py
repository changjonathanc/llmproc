import json
from pathlib import Path


def load_config(path: str | Path) -> dict[str, str]:
    with open(path) as fp:
        return json.load(fp)
