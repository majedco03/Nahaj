"""LLM configuration for the agents."""

import os

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

 
class config:
    def __init__(self):
        load_dotenv()

        api_key = os.getenv("OPENROUTER_API_KEY")
        model = os.getenv("OPENROUTER_MODEL")
        if not api_key or not model:
            raise ValueError("OPENROUTER_API_KEY and OPENROUTER_MODEL are required.")

        self.OPENROUTER_API_KEY = api_key
        self.llm = ChatOpenRouter(
            api_key=api_key,
            model=model,
            temperature=float(os.getenv("AGENT_TEMPERATURE", "0")),
            timeout=int(os.getenv("AGENT_TIMEOUT_SECONDS", "45")),
            max_retries=int(os.getenv("AGENT_MAX_RETRIES", "2")),
        )
