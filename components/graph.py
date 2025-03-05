from components.prompt_generator import generate_prompt
from functools import lru_cache
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph, MessagesState
from sentence_transformers import SentenceTransformer
from utils.utils import get_image_embedding

import numpy as np

@lru_cache()
def build_graph(llm, vector_store):
    def retrieve(state: MessagesState):
        """Retrieve information related to a query."""
        user_message = state["messages"][-1]
        message_text = user_message.content
        print(f"user query: {message_text}")

        embedding_model = SentenceTransformer("clip-ViT-B-32")

        if user_message.additional_kwargs.get('image'):
            image_embedding = get_image_embedding(embedding_model, image_base64=user_message.additional_kwargs.get('image'))
        else:
            image_embedding = np.zeros((1, 512), dtype=np.float32)
        text_embedding = embedding_model.encode(message_text)
        combinted_embedding = np.concatenate([text_embedding, image_embedding]).flatten().tolist()

        retrieved_docs = vector_store.similarity_search_by_vector(combinted_embedding)

        print("Retrieved Docs:", retrieved_docs)
        serialized = "\n\n".join(
            (f"Content: {doc.page_content}") for doc in retrieved_docs
        )
        state["messages"].append(SystemMessage(serialized))
        return state

    def generate(state: MessagesState):
        """Generate answer."""
        conversation_messages = [ message for message in state["messages"] if message.type in ('human', 'ai')]
        last_user_message = [message for message in conversation_messages if message.type == 'human'][-1]
        last_retrieved_message = [message for message in state["messages"] if message.type == 'system'][-1]

        message_text = "\n Give me the list of products from the retrived_documents"
        message_content = generate_prompt().format(user_input=last_user_message.content + message_text, retrieved_documents=last_retrieved_message.content)
        # Run
        prompt = conversation_messages[:-1] + [message_content]
        print(f"Completed Prompt to LLM: {prompt}")
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
