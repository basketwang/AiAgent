from components.prompt_generator import generate_prompt
from cores.model_factory import ModelFactory, get_model_factory
from cores.store_factory import StoreFactory, VectorStoreType, get_store_factory
from fastapi import Depends
from functools import lru_cache
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph, MessagesState
from sentence_transformers import SentenceTransformer
from utils.utils import get_image_embedding

import numpy as np

@lru_cache()
def build_graph(model:str, model_provider:str, vector_store_type: VectorStoreType):
    def retrieve(state: MessagesState):
        """Retrieve information related to a query."""
        user_message = state["messages"][-1]
        message_text = user_message.content
        print(f"user query: {message_text}")

        embedding_model = SentenceTransformer("clip-ViT-B-32")
        text_embedding = embedding_model.encode(str(message_text))
        text_embedding = np.asarray(text_embedding, dtype=np.float32)

        image_base64 = user_message.additional_kwargs.get('image')
        image_embedding = get_image_embedding(embedding_model, image_base64=image_base64) if image_base64 else np.zeros(512, dtype=np.float32)
        image_embedding = np.asarray(image_embedding, dtype=np.float32)
        combined_embedding = np.concatenate([text_embedding, image_embedding]).flatten().tolist()

        vector_store = get_store_factory().get_vector_store(vector_store_type)
        retrieved_docs = vector_store.similarity_search_by_vector(combined_embedding)

        print("Retrieved Docs:", retrieved_docs)
        serialized = "\n\n".join((f"Content: {doc.page_content}") for doc in retrieved_docs)
        state["messages"].append(SystemMessage(serialized))
        return state

    def generate(state: MessagesState):
        """Generate answer."""
        conversation_messages = [ message for message in state["messages"] if message.type in ('human', 'ai')]
        last_user_message = next((message for message in reversed(conversation_messages) if message.type == 'human'), None)
        last_retrieved_message = next((message for message in reversed(state["messages"]) if message.type == 'system'), None)

        if not last_user_message or not last_retrieved_message:
            raise ValueError("Missing required 'human' or 'system' messages.")

        if last_user_message.additional_kwargs.get('image'):
            message_text = f"{last_user_message.content}\n\n The picture has been used to find relavent products, just show the list from the retrieved_documents"
        else:
            message_text = last_user_message.content
        message_content = generate_prompt().format(user_input=message_text, retrieved_documents=last_retrieved_message.content)
        # Run
        prompt = conversation_messages[:-1] + [message_content]
        print(f"Completed Prompt to LLM: {prompt}")
        llm = get_model_factory().get_llm_model(model, model_provider)
        print(llm)
        response = llm.invoke(prompt)
        state["messages"].append(response)
        return state

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(retrieve)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("retrieve")
    graph_builder.add_edge("retrieve","generate")
    graph_builder.add_edge("generate", END)
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph
