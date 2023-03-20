from emergent import ChatAgent

agent = ChatAgent()

while True:
    message = input("You: ")
    response = agent.send(message)
    print("Agent:", response)

