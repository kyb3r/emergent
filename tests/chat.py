from emergent import ChatAgent, HierarchicalMemory

memory = HierarchicalMemory()
agent = ChatAgent(memory=memory)

while True:
    message = input("You: ")
    if message == "quit":
        break
    response = agent.send(message)
    print("Agent:", response)


print(memory.to_json())
