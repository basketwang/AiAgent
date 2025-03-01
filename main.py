from contextlib import asynccontextmanager
from components.prompt_generator import generate_prompt
from cores.model_factory import ModelFactory
from fastapi import FastAPI, Depends
from fastapi.responses import HTMLResponse
from functools import lru_cache
from langchain.schema import HumanMessage
from messages.message import MessageCreateRequest
from rags.data_loader import load_data
from rags.store_factory import StoreFactory, get_store_factory
from rags.graph import build_graph

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("App is starting...")
    load_data()

    yield
    # Shutdown logic
    print("App is shutting down...")

@lru_cache()
def get_model_factory() -> ModelFactory:
    return ModelFactory()

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Chat App</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 2em; }
                #chat-box { border: 1px solid #ccc; padding: 10px; height: 300px; overflow-y: scroll; margin-bottom: 10px; }
                .message { margin: 5px 0; }
                .user { color: blue; }
                .bot { color: green; }
            </style>
        </head>
        <body>
            <h2>Chat App</h2>
            <div id="chat-box">
                <!-- Chat history will be dynamically inserted here -->
            </div>
            <input type="text" id="message-input" placeholder="Type your message...">
            <button onclick="sendMessage()">Send</button>

            <script>
                // Function to Send Message to create_message Endpoint
                async function sendMessage() {
                    const messageInput = document.getElementById("message-input");
                    const message = messageInput.value;
                    if (message.trim() === "") return;

                    // Send POST request to /create_message
                    const response = await fetch("/create_message", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({
                            "message": message
                        })
                    });

                    // Get response JSON and update chat box
                    const data = await response.json();
                    console.log(data)
                    updateChatBox(message, data.message);
                    messageInput.value = "";  // Clear input
                }
                // Function to Update Chat Box
                function updateChatBox(userMessage, botReply) {
                    const chatBox = document.getElementById("chat-box");
                    chatBox.innerHTML += `<div class="message user"><strong>You:</strong> ${userMessage}</div>`;
                    chatBox.innerHTML += `<div class="message bot"><strong>Bot:</strong> ${botReply}</div>`;
                    chatBox.scrollTop = chatBox.scrollHeight;  // Auto-scroll to bottom
                }
           </script>
        </body>
    </html>
   """

@app.post("/create_message")
def create_message(payload: MessageCreateRequest, model_factory: ModelFactory = Depends(get_model_factory), store_factory: StoreFactory = Depends(get_store_factory)):
    llm = model_factory.get_groq_model()
    vector_store = store_factory.get_in_memory_store()
    prompt = generate_prompt() 

    graph = build_graph(prompt, llm, vector_store)

    response = graph.invoke({"question": f"{payload.message}"})
    # Example processing logic (e.g., save to database)
    return {
        "message": response['answer']
    }


