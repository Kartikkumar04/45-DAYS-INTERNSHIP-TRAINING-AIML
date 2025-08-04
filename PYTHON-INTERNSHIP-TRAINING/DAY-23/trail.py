import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load .env file
load_dotenv()

# Set OpenRouter API
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://api.openrouter.ai/v1"

def get_response(user_message):
    chat = ChatOpenAI(
        model="openai/gpt-3.5-turbo",  # or "mistralai/mistral-7b-instruct"
        temperature=0.7
    )
    response = chat([HumanMessage(content=user_message)])
    return response.content
