import emergent
from emergent import ChatAgent
from langchain import SerpAPIWrapper


@emergent.tool()
def google_search(query):
    """Search the web using this tool."""
    serpapi = SerpAPIWrapper()
    return serpapi.run(query)


agent = ChatAgent(tools=[google_search], model="gpt-4")

agent.run()
