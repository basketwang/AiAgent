from cores.store_factory import  get_store_factory, VectorStoreType
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

import json
import os

IMAGE_BASE_DIR = "./static/images/"
def load_data():
    with open("./apple_products.json", "r") as f:
        product_data = json.load(f)

    image_model = SentenceTransformer("clip-ViT-B-32")
    documents = []
    for product in product_data["products"]:
        documents.extend(process_product(product, image_model))
    store = get_store_factory().get_vector_store(VectorStoreType.FAISS, documents)
    store.save_local("faiss_index")
    print(f"Finish loading data")


def process_product(product, image_model):
    documents = []
    for config in product["configurations"]:
        storage = config.get("storage", "N/A")
        ram = config.get("ram", "N/A")
        chip = config.get("chip", "N/A")

        for option in config["options"]:
            color = option["color"]
            price = option["price"]
            image_path = option["image"]
            full_image_path = os.path.join(IMAGE_BASE_DIR, os.path.basename(image_path))
            text_description = f"{product['name']} - {storage} {ram} {chip}, Color: {color}, Price: ${price}"

            # Store both in the vector store
            metadata = {
                "name": product["name"],
                "storage": storage,
                "ram": ram,
                "chip": chip,
                "color": color,
                "price": price,
                "image_path": image_path,
                "description": text_description,
                "full_image_path": full_image_path
            }

            documents.append(Document(page_content=text_description, metadata=metadata))
    return documents
