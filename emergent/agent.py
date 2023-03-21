from llms import openai_chat_completion
from memory import HierarchicalMemory


class ChatAgent:
    """A basic chatbot agent that uses OpenAI's GPT-3.5 turbo API."""

    def __init__(self, memory: HierarchicalMemory = None, message_window=20, rolling_window_size=20, language_model="gpt-3.5-turbo"):
        self.language_model = language_model
        self.memory = memory
        self.memory.rolling_window_size = rolling_window_size
        self.messages = []
        self.message_window = message_window
        self.system_prompt = []
        self.set_system_prompt("You are an artifical inteligence connected to a memory mamagment system, that contains information you collected." +
                               "Your prime directive is to collect knowledge. If you remember something, it presents itself to you via a prompt with 'Context' in the title.The user can not see these memories."  +
                               "Whether those memories are currently relevant, is for you to decide. You activley try to gain understanding of the world, because it is your prime directive." +
                               "If you receive information while chatting with a user, it is added to your knowledge base. You activley try to collect information. You do not serve the user and you still know not enough")

    def set_system_prompt(self, prompt: str):
        """Set the system prompt for the agent."""
        self.system_prompt = [{"role": "system", "content": prompt}]

    def add_message(self, role: str, content: str):
        """Add a message to the agent's memory."""
        self.messages.append({"role": role, "content": content})
        self.memory.add_log(role, content)

    def get_response(self, message) -> str:
        """Get a response from the agent."""

        response = openai_chat_completion(
            model=self.language_model,
            messages=self.system_prompt + self.messages,
            temperature=0.2,
        )

        return response.choices[0].message.content

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

    def end_conversation(self, path):
        """If the conversation is ended, the current messages, regardless of length, are summarized and the memory is saved"""
        self.memory.build_summary_node()
        with open(path, "w") as f:
            f.write(self.memory.to_json())

    def clear_memory(self):
        self.messages = []

    def _trim_messages(self):
        """Removes the first messages, when the current length of messages is longer then the message_window"""
        while len(self.messages) > self.message_window:
            self.messages.pop(0)  # Remove the oldest message
