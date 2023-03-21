from .llms import openai_chat_completion
from .memory import HierarchicalMemory


class ChatAgent:
    """A basic chatbot agent that uses OpenAI's GPT-3.5 turbo API."""

    def __init__(self, memory: HierarchicalMemory = None):
        self.memory = memory
        self.messages = []
        self.set_system_prompt("Never say 'based on the context provided' or 'previous information'. You are David, an LLM that has access to long term memories. Memories are provided to you in messages that have 'Context' in the title. You can use these memories to help you answer questions. The user can not see your memories. ")

    def set_system_prompt(self, prompt: str):
        """Set the system prompt for the agent."""
        self.messages.append({"role": "system", "content": prompt})

    def add_message(self, role: str, content: str):
        """Add a message to the agent's memory."""
        self.messages.append({"role": role, "content": content})
        self.memory.add_log(role, content)
        with open("memories.json", "w") as f:
            f.write(self.memory.to_json())

    def get_response(self, message) -> str:
        """Get a response from the agent."""

        response = openai_chat_completion(
            model="gpt-3.5-turbo",
            messages=self.messages,
            temperature=0.2,
        )

        content = response.choices[0].message.content
        return content

    def send(self, message) -> str:
        """Send a message to the agent. While also managing chat history."""
        self.add_message(role="user", content=message)
        context = self.memory.query(query=message)
        if context:
            self.messages.append(dict(
                role="assistant",
                content="Thought: I know the user can't see this\n My previous memories:\n\n```\n"
                + context.content + "\n```",
                # name="assistant_memory_system"
            ))
        response = self.get_response(message)
        self.add_message(role="assistant", content=response)
        return response
