from emergent import ChatAgent, HierarchicalMemory
import emergent
from colorama import init, Fore, Style

init(autoreset=True)

memory = HierarchicalMemory.from_json("memories.json")

@emergent.tool()
def search_memory(query):
    """Search through your own memories using this tool."""
    return memory.query(query).content

def print_colored(text, color=Fore.RESET):
    print(color + text, end="", flush=True)

def process_response(response):
    thinking = False
    using_tool = False
    for token in response:
        if (token == "<hidden" or token == "<") and not thinking:
            print_colored("[thinking...]\n", Fore.YELLOW)
            thinking = True
        elif token == "__" and not using_tool:
            print_colored("[using tool...] ", Fore.GREEN)
            using_tool = True
        elif not using_tool and not thinking and isinstance(token, str):
            print_colored(token)

        if isinstance(token, dict):
            if "tool_name" in token:
                print_colored(f'[{token["tool_name"]}] ', Fore.GREEN)
                print_colored(f'args {token["tool_params"]}\n', Fore.LIGHTGREEN_EX)
                using_tool = False
                thinking = False
            if "tool_result" in token:
                using_tool = False
                thinking=False

        if token == '">' and thinking:
            thinking = False

agent = ChatAgent(memory=memory, tools=[search_memory], model="gpt-4")
# print(agent.system_prompt)
# print(agent.k_shot_messages)

agent.memory.logs = []

while True:
    print_colored("You: ", Fore.BLUE)
    message = input("")
    print()
    if message == "quit":
        break
    response = agent.send(message)
    print_colored("Agent: ", Fore.BLUE)
    process_response(response)
    print("\n")

print(agent.messages)
