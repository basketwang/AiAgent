from pydantic import BaseModel, Field

class MessageCreateRequest(BaseModel):
    message: str = Field(..., description="The content of the message")
    image: str | None = Field(None, description="The string of the image")



