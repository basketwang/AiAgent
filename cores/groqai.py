import os
from langchain.chat_models import init_chat_model
from langchain_core.language_models import LanguageModelInput, LanguageModelOutput

class GroqAi:
    def __init__(self) -> None:
        #Groq, gmail
        #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
        if not os.environ.get("GROQ_API_KEY"):
            raise Exception("Missing GROQ_API_KEY")  

        self.llm = init_chat_model("llama3-8b-8192", model_provider="groq")

    def invoke(self, message:LanguageModelInput) -> LanguageModelOutput:
        return self.llm.invoke(message)
