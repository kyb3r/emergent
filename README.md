# Emergent

Emergent is a library designed to enhance LLMs like GPT-4 by providing easy integration of external tools and long-term memory capabilities. With Emergent, you can effortlessly transform any Python function into a tool that LLMs can utilize. The library also includes a built-in implementation of a long-term memory system called [HMCS](https://github.com/daveshap/HierarchicalMemoryConsolidationSystem), which seeks to emulate human-like memory organization and recall. 

# Usage

Making your own tools

```python
import emergent
from emergent import ChatAgent

@emergent.tool()
def search(query):
    """This tool is useful for searching through the company's documents."""
    ...
    return "Results: ..."
   
agent = ChatAgent(tools=[search])
response = agent.send("What is the company's policy on remote work?")
    
```

Using the long-term memory system

```python
from emergent import ChatAgent, HierarchicalMemory

memory = HierarchicalMemory.from_json("memories.json")
agent = ChatAgent(memory=memory)

response = agent.send("What is my birthday?")
```



# Overview

The HMCS consists of several key components and processes:

1. Log-based memory: Records thoughts, inputs, and outputs as individual logs.
2. Rollup summaries: Periodically consolidates logs into higher-level summaries to reduce memory load.
3. KB articles: Organizes rollup summaries based on semantic similarity to create or update knowledge base (KB) articles.
4. Clustering or gating functions: Determines topical boundaries for efficient storage and retrieval of information.
5. Scalability: Adapts memory management to accommodate growing data volumes.
6. Periodic reindexing events: Optimizes memory by reorganizing and pruning the hierarchy as needed.


## Note: This is a work in progress

This project tries to implement this system, here's the progress so far.
- [x] Log based memory
- [x] Rollup summaries 
- [x] KB articles, automatically create or update existing KB articles
- [x] Clustering, or gating functions
- [ ] Scalability
- [ ] Periodic reindexing



# Example conversation

I had a conversation with chatgpt earlier about two different topics. After that conversation. I saved the memory to a json file and wiped chatGPT's context window. When we started a new conversation, ChatGPT demonstrated its ability to recall pertinent details effectively.

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

or for development, clone the repository and then run
```
pip install -e .
```
