from contextlib import asynccontextmanager
from cores.store_factory import VectorStoreType
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.runnables import RunnableConfig
from utils.utils import MessageCreateRequest
from PIL import Image
from components.data_loader import load_data
from components.graph import build_graph
from transformers import BlipProcessor, BlipForConditionalGeneration

import os
import uuid

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("App is starting...")
    load_data()

    yield
    # Shutdown logic
    print("App is shutting down...")

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

session_threads = {}

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
            <input type="file" id="image-input" accept="image/*">

            <script>
                // Function to Send Message to create_message Endpoint
                const user_id = crypto.randomUUID();

                async function sendMessage() {
                    const messageInput = document.getElementById("message-input");
                    const imageInput = document.getElementById("image-input");
                    let message = messageInput.value;
                    const file = imageInput.files[0];
                    if (message.trim() === "" && !file) return;

                    let imageBase64 = null;
                    let formData = new FormData();
                    if (file) {
                        const reader = new FileReader();
                        reader.readAsDataURL(file);
                        reader.onload = async function () {
                            imageBase64 = reader.result.split(",")[1]; // Extract Base64 data
                            await sendJsonRequest(user_id, message, imageBase64);
                        };
                        imageInput.value = '';
                    } else {
                        await sendJsonRequest(user_id, message, null);
                    }
                    messageInput.value = '';
                }
                async function sendJsonRequest(user_id, message, imageBase64) {
                    const payload = {
                        user_id: user_id,
                        message: message,
                        image: imageBase64
                    };

                    // Send POST request to /create_message with JSON
                    const response = await fetch("/create_message", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify(payload)
                    });

                    const data = await response.json();
                    updateChatBox(message, imageBase64, data.message);
                }

                function updateChatBox(userMessage, imageBase64, botReply) {
                    const chatBox = document.getElementById("chat-box");

                    // Append user message
                    if (userMessage) {
                        const userMessageDiv = document.createElement("div");
                        userMessageDiv.classList.add("message", "user");
                        userMessageDiv.innerHTML = `<strong>You:</strong> ${userMessage}`;
                        chatBox.appendChild(userMessageDiv);
                    }

                    // Append image if uploaded
                    if (imageBase64) {
                        const imageDiv = document.createElement("div");
                        imageDiv.classList.add("message", "user");

                        const img = document.createElement("img");
                        img.src = `data:image/png;base64,${imageBase64}`;
                        img.style.maxWidth = "200px";
                        img.style.marginTop = "5px";

                        imageDiv.appendChild(img);
                        chatBox.appendChild(imageDiv);
                    }

                    // Append bot response
                    const botMessageDiv = document.createElement("div");
                    botMessageDiv.classList.add("message", "bot");
                    botMessageDiv.innerHTML = `<strong>Bot:</strong> ${botReply}`;
                    chatBox.appendChild(botMessageDiv);

                    // Auto-scroll
                    chatBox.scrollTop = chatBox.scrollHeight;
                }
           </script>
        </body>
    </html>
   """

@app.post("/create_message")
def create_message(payload: MessageCreateRequest):
    graph = build_graph("llama3-8b-8192", "groq", VectorStoreType.FAISS)

    input = {"messages": [{"type": "human", "content": payload.message, "image": payload.image}]}
    response = graph.invoke(input, config=RunnableConfig(configurable={"thread_id": payload.user_id}))
    print(payload.user_id)
    return {
        "message": response['messages'][-1].content
    }

#A test to recognize images. Could potentially used to construct prompt for image-based searching
@app.get("/test_image_to_text")
def test_image_to_text():
    image_path = os.path.join("./static/images/", "iphone_pro_black.jpg")
    image = Image.open(image_path)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    inputs = processor(images=image, return_tensors="pt")

    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    print(caption)
    return {
        "message": caption
    }

