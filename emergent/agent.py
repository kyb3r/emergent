from .llms import openai_chat_completion
from .memory import HierarchicalMemory


class ChatAgent:
    """A basic chatbot agent that uses OpenAI's GPT-3.5 turbo API."""

    def __init__(self, memory: HierarchicalMemory = None):
        self.memory = memory
        self.messages = []
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt for the agent."""
        self.messages.append({"role": "assistant", "content": prompt})

    def add_message(self, role: str, content: str):
        """Add a message to the agent's memory."""
        self.messages.append({"role": role, "content": content})

        if self.memory:
            raise NotImplementedError

    def get_response(self, message) -> str:
        """Get a response from the agent."""

        response = openai_chat_completion(
            model="gpt-3.5-turbo",
            messages=self.messages,
            temperature=0.5,
        )

        content = response.choices[0].message.content
        return content

    def send(self, message) -> str:
        """Send a message to the agent. While also managing chat history."""
        self.add_message(role="user", content=message)
        response = self.get_response(message)
        self.add_message(role="assistant", content=response)
        return response
