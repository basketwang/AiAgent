from functools import lru_cache
from rags.in_memory_store import InMemoryStore
from threading import Lock

class StoreFactory:
    def __init__(self):
        self._lock = Lock()
        self.in_memory_store = None

    def get_in_memory_store(self):
        # Double locking for thread safety
        if not self.in_memory_store:
            with self._lock:
                if not self.in_memory_store:
                    self.in_memory_store = InMemoryStore()
        return self.in_memory_store

@lru_cache()
def get_store_factory() -> StoreFactory:
    return StoreFactory()


