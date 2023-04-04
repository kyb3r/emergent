from emergent import ChatAgent, HierarchicalMemory
import emergent
from colorama import init, Fore, Style

memory = HierarchicalMemory.from_json("memories.json")


@emergent.tool()
def search_memory(query):
    """Search through your own memories using this tool."""
    return memory.query(query).content


agent = ChatAgent(memory=memory, tools=[search_memory], model="gpt-3.5-turbo")
agent.run()
