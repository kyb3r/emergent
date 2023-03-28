from emergent import ChatAgent, HierarchicalMemory
import emergent


memory = HierarchicalMemory.from_json("memories.json")

@emergent.tool()
def search_memory(query):
    """Search through your own memories using this tool."""
    return memory.query(query).content

agent = ChatAgent(memory=memory, tools=[search_memory], model="gpt-4")
print(agent.system_prompt[0]["content"])

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
