from enum import Enum
from functools import lru_cache
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from sentence_transformers import SentenceTransformer
from threading import Lock
from utils.utils import get_image_embedding
import faiss
import numpy as np

class VectorStoreType(Enum):
    IN_MEMORY = 1
    FAISS = 2

class StoreFactory:
    def __init__(self):
        self._lock = Lock()
        self.store_pool = {}

    def get_vector_store(self, store_type:VectorStoreType, documents=None):
        with self._lock:
            if store_type not in self.store_pool:
                if store_type == VectorStoreType.IN_MEMORY:
                    self.store_pool[store_type] = self.create_in_memory_store(documents)
                elif store_type == VectorStoreType.FAISS:
                    self.store_pool[store_type] = self.create_faiss_store(documents)
 
            return self.store_pool[store_type]


    def create_in_memory_store(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(documents)
        vector_store = InMemoryVectorStore(DeterministicFakeEmbedding(size=4096))
        vector_store.add_documents(documents=all_splits)
        return vector_store

    def create_faiss_store(self, documents=None):
        if not documents:
            raise ValueError("FAISS requires documents with embeddings to initialize!")

        embedding_model = SentenceTransformer("clip-ViT-B-32")
        index = faiss.IndexFlatL2(1024)
        faiss_store = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        embeddings = []
        for i, document in enumerate(documents):
            text_embedding = embedding_model.encode(document.page_content)
            image_embedding = get_image_embedding(embedding_model, image_path=document.metadata["full_image_path"])
            text_embedding = np.array(text_embedding, dtype=np.float32)
            image_embedding = np.array(image_embedding, dtype=np.float32)
            combined_embedding = np.concatenate([text_embedding, image_embedding])
            embeddings.append(combined_embedding)

        embeddings = np.array(embeddings, dtype="float32")
        faiss_store.index.add(embeddings)
        for i, document in enumerate(documents):
            faiss_store.docstore.add({str(i): document})
            faiss_store.index_to_docstore_id[i] = str(i)

        return faiss_store

@lru_cache()
def get_store_factory() -> StoreFactory:
    return StoreFactory()


