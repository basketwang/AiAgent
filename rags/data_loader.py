from fastapi import Depends
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rags.store_factory import StoreFactory, get_store_factory

def load_data():
    file_path='./apple_products.json'
    jq_schema = ".products[] | {name: .name, storage: .configurations[].storage, color: .configurations[].options[].color, price: .configurations[].options[].price}"

    loader = JSONLoader(
         file_path=file_path,
         jq_schema=jq_schema,
         text_content=False)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    store = get_store_factory().get_in_memory_store()
    _ = store.add_documents(documents=all_splits)
