import emergent
from emergent import ChatAgent
from langchain import SerpAPIWrapper


@emergent.tool()
def search_web(query):
    """Search the web using this tool."""
    serpapi = SerpAPIWrapper()
    return serpapi.run(query)

agent = ChatAgent(tools=[search_web], model="gpt-4")

print(agent.system_prompt)
agent.run()