from cores.groqai import GroqAi
from threading import Lock

class ModelFactory:
    def __init__(self):
        self._lock = Lock()
        self.groq = None


    def get_groq_model(self):
        # Double locking for thread safety
        if not self.groq:
            with self._lock:
                if not self.groq:
                    self.groq = GroqAi()
        return self.groq
