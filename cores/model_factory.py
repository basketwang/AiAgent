from functools import lru_cache
from langchain.chat_models import init_chat_model
from threading import Lock

class ModelFactory:
    def __init__(self):
        self._lock = Lock()
        self.llm_store = {}

    def get_llm_model(self, model:str, model_provider:str):
        key = ':'.join([model_provider, model])
        with self._lock:
            if key not in self.llm_store:
                self.llm_store[key] = init_chat_model(model, model_provider=model_provider)
            return self.llm_store[key]

@lru_cache()
def get_model_factory() -> ModelFactory:
    return ModelFactory()
