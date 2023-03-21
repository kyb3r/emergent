# Emergent

Emergent is an implementation of the Hierarchical Memory Consolidation System ([HMCS](https://github.com/daveshap/HierarchicalMemoryConsolidationSystem)) for large language models like ChatGPT. HMCS aims to mimic human-like memory organization and recall, enabling efficient storage and retrieval of information while adapting to growing data.

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
- [x] Clustering, or gating functions (We use an LLM to make a classifcation whether or not a summary is relevant to the KB article as a gating function)
- [ ] Scalability
- [ ] Periodic reindexing


# Usage

```python
from emergent import ChatAgent, HierarchicalMemory

memory = HierarchicalMemory.from_json("memories.json")
agent = ChatAgent(memory=memory)

response = agent.send("What is my birthday?")
```

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
