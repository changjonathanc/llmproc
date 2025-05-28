import json
from pathlib import Path
from functools import lru_cache
@lru_cache(maxsize=32)
def load_config(path: str | Path) -> dict[str, str]:
    with open(path) as fp:
        return json.load(fp)
