from io import BytesIO
from PIL import Image
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import Optional

import base64
import os

class MessageCreateRequest(BaseModel):
    user_id: str = Field(..., description="user_id")
    message: str = Field(..., description="The content of the message")
    image: str | None = Field(None, description="The string of the image")

def get_image_embedding(embedding_model:SentenceTransformer, image_path:Optional[str] = None, image_base64:Optional[str] = None):
    if image_path:
        if os.path.exists(image_path):
            image = Image.open(image_path)
            return embedding_model.encode(image)
        else:
            print(f"Warning: Image {image_path} not found!")
            return None
    elif image_base64:
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_bytes))
            return embedding_model.encode(image)
    else:
        raise Exception("Either image_path or image_base64 has to be provided")


