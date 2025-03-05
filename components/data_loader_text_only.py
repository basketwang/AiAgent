from langchain_community.document_loaders import JSONLoader
from cores.store_factory import  get_store_factory, VectorStoreType

# Used at the first stage to process text only agent. Doen't support Image
def load_data():
    file_path='./apple_products.json'
    jq_schema = ".products[] | {name: .name, storage: .configurations[].storage, color: .configurations[].options[].color, price: .configurations[].options[].price}"

    loader = JSONLoader(
         file_path=file_path,
         jq_schema=jq_schema,
         text_content=False)
    docs = loader.load()
    get_store_factory().get_vector_store(VectorStoreType.IN_MEMORY, docs)
