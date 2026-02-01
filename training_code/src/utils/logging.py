import json
import os
import time
from typing import Any, Dict


class JsonlLogger:
    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._fp = open(path, "a", encoding="utf-8")

    def log(self, data: Dict[str, Any]) -> None:
        payload = {"time": time.time(), **data}
        self._fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()
