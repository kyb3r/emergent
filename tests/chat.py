from emergent import ChatAgent, HierarchicalMemory

from openai.embeddings_utils import get_embedding

memory = HierarchicalMemory.from_json("memories.json")

print("Loaded memories")

agent = ChatAgent(memory=memory)
agent.memory.logs = []

while True:
    message = input("You: ")
    if message == "quit":
        break
    response = agent.send(message)
    print("Agent:", response)

print(agent.messages)

# with open("memories.json", "w") as f:
#     f.write(memory.to_json())

# print(memory.to_json())
