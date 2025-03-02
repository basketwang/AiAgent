from components.prompt_generator import generate_prompt
from functools import lru_cache
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import List, TypedDict

@lru_cache()
def build_graph(llm, vector_store):
    def retrieve(state: MessagesState):
        """Retrieve information related to a query."""
        user_message = state["messages"][-1]
        query = user_message.content
        print(f"user query: {query}")

        retrieved_docs = vector_store.similarity_search(query)
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

        message_content = generate_prompt().format(user_input=last_user_message.content, retrieved_documents=last_retrieved_message.content)
        # Run
        prompt = conversation_messages[:-1] + [message_content]
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
