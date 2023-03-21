import uuid
from typing import List, Dict
from collections import defaultdict
from datetime import datetime
import json
import logging
from openai.embeddings_utils import cosine_similarity
from llms import chat_gpt_prompt, get_embedding


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)


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
        self.embedding = None

    @chat_gpt_prompt
    def _summary_prompt(self) -> str:
        """
        Generates a summary of the memory logs. ChatGPT executes the prompt
        returned by this method.
        """

        prompt = "The following is a conversation between you and the user:\n\n"

        for log in self.logs:
            prompt += f"{log.role.capitalize()}: {log.content}\n\n"

        prompt += "Based on the above conversation, write an executive summary with only the most important information. Make sure to include important things like user information, ideas, events, etc.:"

        return prompt

    def generate_summary(self) -> str:
        """Generate a summary of the memory logs."""
        self.content = self._summary_prompt()
        self.embedding = get_embedding(self.content)

    def to_dict(self) -> Dict:
        return {
            "id": str(self.id),
            "logs": [log.to_dict() for log in self.logs],
            "content": self.content,
            "embedding": self.embedding,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: Dict):
        logs = [MemoryLog.from_dict(log_data) for log_data in data["logs"]]
        summary_node = SummaryNode(logs=logs)
        summary_node.id = uuid.UUID(data["id"])
        summary_node.content = data["content"]
        summary_node.created_at = data["created_at"]
        summary_node.embedding = data["embedding"]
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
        self.embedding = None

    @chat_gpt_prompt
    def _article_prompt(self):
        """
        Generates a knowledge base article based on the summary nodes. ChatGPT
        executes the prompt returned by this method.
        """

        prompt = "The following is a collection of summaries of chat logs:\n\n"

        for index, summary_node in enumerate(self.summary_nodes):
            prompt += f"{index+1}. {summary_node.content}\n\n"

        prompt += (
            "Write a concise summary about the most important details that you can extract from the summary"
            "You must have a title that represent the topic of the article. "
            "The title should categorise this summary as either related to "
            "A person or name, Places, Things, Events, or Concepts. \n"
            "An example title would be: 'Personal information about the user' and the body would contain important details "
        )

        return prompt

    def _update_article_prompt(self, new_summary_node):
        # TODO: make a more advanced version of this
        return self._article_prompt()

    def generate_article(self) -> str:
        self.content = self._article_prompt()
        logging.info(f"<>{self.content}<>")
        self.embedding = get_embedding(self.content)

    def update_article(self, summary_node):
        self.content = self._update_article_prompt(summary_node)
        self.embedding = get_embedding(self.content)

    def to_dict(self) -> Dict:
        return {
            "id": str(self.id),
            "summary_nodes": [cluster.to_dict() for cluster in self.summary_nodes],
            "content": self.content,
            "embedding": self.embedding,
        }

    @staticmethod
    def from_dict(data: Dict):
        summary_nodes = [
            SummaryNode.from_dict(cluster_data)
            for cluster_data in data["summary_nodes"]
        ]
        knowledge_node = KnowledgeNode(summary_nodes=summary_nodes)
        knowledge_node.id = uuid.UUID(data["id"])
        knowledge_node.content = data["content"]
        knowledge_node.embedding = data["embedding"]
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
        self.rolling_window_size = 20

    def query(self, query: str) -> KnowledgeNode:
        """
        This method is responsible for querying the memory for a given query.
        """
        query_embedding = get_embedding(query)
        # find the most similar knowledge node
        if self.knowledge_nodes:
            knowledge_node = max(
                self.knowledge_nodes,
                key=lambda node: cosine_similarity(node.embedding, query_embedding),
            )
            return knowledge_node
        return None

    def add_log(self, role, content) -> None:
        log = MemoryLog(role=role, content=content)
        self.logs.append(log)
        if len(self.logs) == self.rolling_window_size:
            self.build_summary_node()

    @chat_gpt_prompt
    def _llm_classification(self, summary_node, knowledge_node):
        """
        This method is responsible for classifying a summary node as either
        a new knowledge node or an existing knowledge node.
        """
        prompt = (
            f"Given the following summary:\n\n{summary_node.content}\n\n"
            f"and the following knowledge base article:\n\n{knowledge_node.content}\n\n"
            "Please classify whether the summary has relevent information that can be added to the knowledge base article.\n\n"
            "If the summary is not related or relevant to the knowledge base article, please answer with `<no>`\n\n"
            "If the summary is relevant to the knowledge base article, please answer with `<yes>`\n\n"
        )

        return prompt

    def _semantic_similarity(self, summary_node):
        """
        This method is responsible for calculating the semantic similarity
        between a summary node and the knowledge nodes.
        """
        if len(self.knowledge_nodes) == 0:
            return None

        embedding = summary_node.embedding

        similarities = []

        for knowledge_node in self.knowledge_nodes:
            similarity = cosine_similarity(embedding, knowledge_node.embedding)
            similarities.append([similarity, knowledge_node])

        most_similar = max(similarities, key=lambda x: x[0])
        knowledge_node = most_similar[1]

        if "<yes>" in self._llm_classification(summary_node, knowledge_node):
            return knowledge_node
        else:
            return None

    def build_summary_node(self) -> None:
        """After a rolling window of X logs, we build a summary node that summarizes the logs"""
        summary_node = SummaryNode(self.logs)
        summary_node.generate_summary()
        logging.info("<created summary node>")
        self.summary_nodes.append(summary_node)
        self.logs = []

        # If there are no knowledge nodes, we create one with the summary node
        # If there are knowledge nodes, we check if the summary node is similar to any of them
        # If it is, we update the closest knowledge base article with the summary node
        similar_knowledge_node = self._semantic_similarity(summary_node)

        if similar_knowledge_node is not None:
            similar_knowledge_node.summary_nodes.append(summary_node)
            similar_knowledge_node.update_article(summary_node)
            logging.info("<updated knowledge node>")
            logging.info(f"<> {similar_knowledge_node.content} <>")
        else:
            logging.info("<creating new knowledge node>")
            knowledge_node = KnowledgeNode(summary_nodes=[summary_node])
            knowledge_node.generate_article()
            self.knowledge_nodes.append(knowledge_node)
        # If it is not, we create a new knowledge base article with the summary node

    def reindex_knowledge_nodes(self) -> None:
        """
        Clusters summary nodes into different groups, and then uses llm to
        come up with a knowledge base article for each group.
        """
        raise NotImplementedError

    def to_json(self) -> str:
        return json.dumps(
            {
                "logs": [log.to_dict() for log in self.logs],
                "summary_nodes": [
                    summary_node.to_dict() for summary_node in self.summary_nodes
                ],
                "knowledge_nodes": [
                    knowledge_node.to_dict() for knowledge_node in self.knowledge_nodes
                ],
            },
            indent=4,
            cls=DateTimeEncoder,
        )
    
    @classmethod
    def from_json(cls, path: str):
        with open(path) as f:
            data = json.load(f)

        memory = cls()
        memory.logs = [MemoryLog.from_dict(log_data) for log_data in data["logs"]]
        memory.summary_nodes = [
            SummaryNode.from_dict(summary_node_data)
            for summary_node_data in data["summary_nodes"]
        ]
        memory.knowledge_nodes = [
            KnowledgeNode.from_dict(knowledge_node_data)
            for knowledge_node_data in data["knowledge_nodes"]
        ]
        return memory
