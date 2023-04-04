import logging
from emergent.agent import ChatAgent, HierarchicalMemory
import emergent
from openai.embeddings_utils import get_embedding
import openai

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
memory_path = "memories.json"

try:
    memory = HierarchicalMemory.from_json(memory_path)
    logging.info(f"Loaded memories from {memory_path}")
except:
    logging.info(f"No {memory_path} file found, initializing new database")
    memory = HierarchicalMemory(model="gpt-3.5-turbo")

memory.model = "gpt-3.5-turbo"

@emergent.tool()
def search_memory(query):
    """Search through your own memories using this tool."""
    return memory.query(query).content

agent = ChatAgent(memory=memory, tools=[search_memory], model="gpt-4")
agent.memory.logs = []

print(agent.system_prompt)
print(agent.language_model)

name = "User: "

while True:
    try:
        message = input(f"{name}: ")
        if message == "quit":
            break
        response = agent.send(message)
        print("Agent:", response)
    except:
        logging.warning(f"Call failed")
        agent.end_conversation(memory_path)
        break

agent.end_conversation(memory_path)
