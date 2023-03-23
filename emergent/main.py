from emergent import ChatAgent, HierarchicalMemory

import logging
from openai.embeddings_utils import get_embedding
import config
import openai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
openai.api_key = config.OPENAI_API_KEY # add your api key here
memory_path = "memories.json"

try:
    memory = HierarchicalMemory.from_json(memory_path)
    logging.info(f"Loaded memories from {memory_path}")
except:
    logging.info(f"No {memory_path} file found, initializing new database")
    memory = HierarchicalMemory()

agent = ChatAgent(memory=memory, language_model="gpt-4")
agent.memory.logs = []

name = input("Please enter your name: ")
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

print(agent.messages)
agent.end_conversation(memory_path)
