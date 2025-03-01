from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore

class InMemoryStore:
    def __init__(self) -> None:
        # Use Fake embeddings model for the prototype
        embeddings = DeterministicFakeEmbedding(size=4096)
        self.in_memory_store = InMemoryVectorStore(embeddings)

    def add_documents(self, documents) -> None:
        self.in_memory_store.add_documents(documents)

    def similarity_search(self, query:str):
        return self.in_memory_store.similarity_search(query)
