# Emergent

Emergent is an implementation of the Hierarchical Memory Consolidation System ([HMCS](https://github.com/daveshap/HierarchicalMemoryConsolidationSystem)) for large language models like ChatGPT. HMCS aims to mimic human-like memory organization and recall, enabling efficient storage and retrieval of information while adapting to growing data.

Note: This is a work in progress

# Usage

```python
from emergent import ChatAgent, HierarchicalMemory

memory = HierarchicalMemory()
agent = ChatAgent(memory=memory)

response = agent.send("What is my birthday?")
```

# Example conversation

I had a conversation with chatgpt earlier, I introduced it to myself as "Bob" and gave it some personal details, like hometown and age. The memory system automatically created a knowledge base article about me. Later on, I started talking to it about HMCS and I gave it a simple explanation. The memory system determined that this new stuff that we are talking about is different from the personal details we talked about earlier. So it created a new knowledge base article. And whenever I gave new information about the same topic, it updated its memory to have a new summary of the topic. 

After that conversation. I saved the memory to a json file and wiped chatGPT's context window. I had a new conversation and it seem's like it can remember relevant details well. 

```
You: Do you remember me David, I'm bob

Agent: Yes, Bob. It's good to hear from you again. How can I assist you today?

You: Prove you remember me

Agent: Of course, Bob. We have interacted before, and based on the information I have in my memory, you are a 25-year-old resident of [redacted] with four siblings. Is there anything else I can help you with?

You: what is HMCS? Explain it to me

Agent: HMCS stands for Hierarchical Memory Consolidation System. It is a system that is being developed to improve the efficiency and effectiveness of large language models like ChatGPT. The system aims to mimic human-like memory organization and recall, enabling efficient storage and retrieval of information while adapting to growing data. The testing process is ongoing, and the user is monitoring the performance of ChatGPT with the implementation of HMCS.

You: Who's the user?

Agent: The user is you, Bob.

```

# Installation
```
pip install git+https://github.com/kyb3r/emergent
```
