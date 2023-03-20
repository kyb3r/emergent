import uuid
from typing import List, Dict
from collections import defaultdict
from datetime import datetime

from .llms import chat_gpt_prompt


class MemoryLog:
    """
    A class that represents the raw data that the agent collects.

    This would be the raw messages that the agent receives from the user
    and also the messages that the agent sends to the user.
    """

    def __init__(self, role: str, content: str):
        self.id = uuid.uuid4()
        self.role = role
        self.content = content
        self.created_at = datetime.now()

    def to_dict(self) -> Dict:
        return {
            "id": str(self.id),
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: Dict):
        memory_log = MemoryLog(role=data["role"], content=data["content"])
        memory_log.id = uuid.UUID(data["id"])
        memory_log.created_at = data["created_at"]
        return memory_log


class SummaryNode:
    """
    This class represents a rollup of memory logs. It is a summary of a list of
    memory logs that are chronologically sequential.
    """

    def __init__(self, logs: List[MemoryLog]):
        self.id = uuid.uuid4()
        self.logs = logs
        self.content: str
        self.created_at = datetime.now()
    
    @chat_gpt_prompt
    def _summary_prompt(self) -> str:
        """
        Generates a summary of the memory logs. ChatGPT executes the prompt 
        returned by this method.
        """

        prompt = "The following is a conversation between you and the user:\n\n"

        for log in self.logs:
            prompt += f"{log.role.capitalize()}: {log.content}\n\n"
        
        prompt += "Based on the above conversation, write an executive summary:"

        return prompt
    
    def generate_summary(self) -> str:
        """Generate a summary of the memory logs."""
        self.content = self._summary_prompt()

    def to_dict(self) -> Dict:
        return {
            "id": str(self.id),
            "logs": [log.to_dict() for log in self.logs],
            "content": self.content,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: Dict):
        logs = [MemoryLog.from_dict(log_data) for log_data in data["logs"]]
        summary_node = SummaryNode(logs=logs)
        summary_node.id = uuid.UUID(data["id"])
        summary_node.content = data["content"]
        summary_node.created_at = data["created_at"]
        return summary_node


class KnowledgeNode:
    """
    A class that represents a knowledge base article, which is a summary
    or representation of a group of (clustered) summary nodes.
    """

    def __init__(self, summary_nodes: List[SummaryNode]):
        self.id = uuid.uuid4()
        self.summary_nodes = summary_nodes
        self.content: str

    def build(self) -> str:
        # Implement a method to turn the summary nodes into a knowledge base article
        raise NotImplementedError

    def to_dict(self) -> Dict:
        return {
            "id": str(self.id),
            "summary_nodes": [cluster.to_dict() for cluster in self.summary_nodes],
            "content": self.content,
        }

    @staticmethod
    def from_dict(data: Dict):
        summary_nodes = [
            SummaryNode.from_dict(cluster_data) for cluster_data in data["summary_nodes"]
        ]
        knowledge_node = KnowledgeNode(summary_nodes=summary_nodes)
        knowledge_node.id = uuid.UUID(data["id"])
        knowledge_node.content = data["content"]
        return knowledge_node


class HierarchicalMemory:
    """
    This class manages the HMCS system. It is responsible for building the
    summary nodes, knowledge nodes and also querying for memories.

    Example
    -------
    >>> memory = HierarchicalMemory()
    >>> node = memory.query("What is the birthday of bob")
    >>> node
    KnowledgeNode(content="Bob is a user that I interacted with on 12/03/2023, Bob's birthday is on 1/1/2000")
    >>> node.summary_nodes[0].logs
    [
        MemoryLog(role="user", content="My name is Bob"), 
        MemoryLog(role="user", content="My birthday is on 1/1/2000")
    ]
    """

    def __init__(self):
        self.logs: list = []
        self.summary_nodes: list = []
        self.knowledge_nodes: list = []

    def query(self, query: str) -> KnowledgeNode:
        """
        This method is responsible for querying the memory for a given query.
        """
        raise NotImplementedError

    def add_log(self, log: MemoryLog) -> None:
        self.logs.append(log)
        if len(self.logs) == 10:
            self.build_summary_node()

    def build_summary_node(self) -> None:
        """After a rolling window of 10 logs, we build a summary node that summarizes the logs"""
        summary_node = SummaryNode(self.logs)
        summary_node.generate_summary()
        self.summary_nodes.append(summary_node)
        self.logs = []

    def build_knowledge_nodes(self) -> None:
        """
        Clusters summary nodes into different groups, and then uses llm to
        come up with a knowledge base article for each group
        """
        raise NotImplementedError


if __name__ == "__main__":
    

    """

    """
