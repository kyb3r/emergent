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

# Installation
```
pip install git+https://github.com/kyb3r/emergent
```
