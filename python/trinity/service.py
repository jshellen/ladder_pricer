from __future__ import annotations

import json
import threading
from collections import OrderedDict
from copy import deepcopy
from typing import Any

from .engine import Engine


class EngineCache:
    """Small in-process cache keyed only by economic model configuration."""

    def __init__(self, max_entries: int = 8):
        self.max_entries = int(max_entries)
        self._items: OrderedDict[str, Engine] = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def _key(config: dict[str, Any]) -> str:
        return json.dumps(config, sort_keys=True, separators=(",", ":"))

    def get(self, config: dict[str, Any]) -> Engine:
        key = self._key(config)
        with self._lock:
            engine = self._items.get(key)
            if engine is None:
                engine = Engine(deepcopy(config))
                self._items[key] = engine
                while len(self._items) > self.max_entries:
                    self._items.popitem(last=False)
            else:
                self._items.move_to_end(key)
            return engine


ENGINES = EngineCache()
