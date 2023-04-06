import uuid
from typing import List, Dict
from collections import defaultdict
from datetime import datetime
import json
import logging
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

from .llms import chat_gpt_prompt, get_embedding, Prompt


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

    def __init__(self, logs: List[MemoryLog], model="gpt-3.5-turbo"):
        self.id = uuid.uuid4()
        self.logs = logs
        self.content: str
        self.created_at = datetime.now()
        self.embedding = None
        self.model = model

    @chat_gpt_prompt
    def _summary_prompt(self) -> str:
        """
        Generates a summary of the memory logs. ChatGPT executes the prompt
        returned by this method.
        """
        system = (
            "You are knowledgeGPT an AI, that extracts knowledge from chat logs. You just extract information regarding topics like people, concepts or events."
            "Example conversation: USER: Hi, my name is John and I like artificial intelligence\n\n ASSISTANT: Hi John, why do you like artificial intelligence?\n\n USER:"
            " Because recently (2023), AI has gotten a lot smarter, especially in nlp\n\n Example extract information: The user is John. "
            "John likes AI, because it has gotten a lot smarter."
            "The year is 2023 or shortly after. AI connected to natural language processing has gotten a lot smarter in 2023."
        )

        prompt = (
            "The following is the conversation between the ASSISTANT and the USER:\n\n"
        )

        for log in self.logs:
            prompt += f"{log.role.capitalize()}: {log.content}\n\n"

        prompt += (
            "TASK: Based on the above conversation, write a concise list of the information that was shared between the USER and the ASSISTANT."
            "Include every piece of knowledge that was shared, regarding persons, concepts or events, do not add any information that did not come up in the conversation."
            "Only return the summary."
        )

        return Prompt(system=system, prompt=prompt, model=self.model)

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
            "model": self.model,
        }

    @staticmethod
    def from_dict(data: Dict):
        logs = [MemoryLog.from_dict(log_data) for log_data in data["logs"]]
        summary_node = SummaryNode(logs=logs)
        summary_node.model = data.get("model", "gpt-3.5-turbo")
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

    def __init__(self, summary_nodes: List[SummaryNode], model="gpt-3.5-turbo"):
        self.id = uuid.uuid4()
        self.summary_nodes = summary_nodes
        self.topic = None
        self.content: str
        self.embedding = None
        self.model = model

    @chat_gpt_prompt
    def topic_prompt(self):
        system = (
            "You are headingGPT, you write short informative headings for knowledge ARTICLES. Examples for headings if the article is about"
            "a person: 'Information about John', a concept: 'Derivatives in math' or a event '2024 presidential election (USA)'."
        )

        prompt = (
            f"ARTICLE: {self.content}\n\n"
            f"TASK: Based on this ARTICLE please write an heading for the ARTICLE. The heading should be informative and capture the essence of the content of the article."
            f"Only return the heading"
        )

        return Prompt(system=system, prompt=prompt, model=self.model)

    def generate_topic(self):
        self.topic = self.topic_prompt()

    @chat_gpt_prompt
    def _article_prompt(self, topic):
        """
        Generates a knowledge base article based on the summary nodes, and the topic provided. ChatGPT
        executes the prompt returned by this method.
        """
        system = (
            "You are knowledgeGPT, you generate knowledge base articles regarding a specific topic. Here an example: "
            "INFORMATION: 'The user is John. John likes AI, because it has gotten a lot smarter."
            "The year is 2023 or shortly after. AI connected to natural language processing has gotten a lot smarter in 2023.' "
            "topic: 'Artificial intelligence' knowledge article: 'There are different fields related to AI. AI connected"
            "to natural language processing (nlp), has gotten a lot smarter in 2023. Some people like AI.'"
        )

        prompt = "INFORMATION: "

        for index, summary_node in enumerate(self.summary_nodes):
            prompt += f"{index + 1}. {summary_node.content}\n\n"

        prompt += (
            f"TASK: Based on this INFORMATION, extract all knowledge that regards the following topic: {topic},"
            f"and write a short knwoledge article about that topic. Be sure to include each piece of knowledge, that directly relates to {topic} and was included in the INFORMATION."
            f"Don't add anything that was not included in the INFORMATION. Return only the knowledge article"
        )

        return Prompt(system=system, prompt=prompt, model=self.model)

    @chat_gpt_prompt
    def _update_article_prompt(self, new_summary_node, topic):
        system = (
            "You are knowledgeGPT, you update knowledge base articles regarding a specific topic. Here an example: "
            "INFORMATION: 'The user is John. John loves playing tennis. The next presidential election in the USA will happen next year."
            "The year is 2023 or shortly after. GPT-4 was released, a transformer in the nlp domain, that greatly advanced the field' "
            "topic: 'Artificial intelligence' old knowledge article: 'There are different fields related to AI. AI connected"
            "to natural language processing (nlp), has gotten a lot smarter in 2023. This could be due to the release of GPT-4 a transformer model "
            "in the nlp domain, that advanced the field. Some people like AI.' updated article:"
            "'There are different fields related to AI. AI connected"
            "to natural language processing (nlp), has gotten a lot smarter in 2023. Some people like AI.'"
        )

        prompt = f"ARTICLE: {self.content}"

        prompt += f"NEW INFORMATION:  {new_summary_node.content}\n\n"

        prompt += (
            f"TASK: Based on this NEW INFORMATION, update the ARTICLE with all information that regards the following topic: {topic}. "
            f"Be sure to include each piece of knowledge, that directly relates to {topic} and was included in the NEW INFORMATION or the ARTICLE."
            f"Don't add anything that was not included in the NEW INFORMATION or the ARTICLE. Return only the updated article"
        )

        return Prompt(system=system, prompt=prompt, model=self.model)

    def generate_article(self, topic):
        self.content = self._article_prompt(topic)
        logging.info(f"<>{self.content}<>")
        self.embedding = get_embedding(self.content)

    def update_article(self, summary_node, topic):
        self.content = self._update_article_prompt(summary_node, topic)
        self.embedding = get_embedding(self.content)

    def to_dict(self) -> Dict:
        return {
            "id": str(self.id),
            "summary_nodes": [node.to_dict() for node in self.summary_nodes],
            "content": self.content,
            "embedding": self.embedding,
            "model": self.model,
            "topic": self.topic,
        }

    @staticmethod
    def from_dict(data: Dict):
        summary_nodes = [
            SummaryNode.from_dict(node_data) for node_data in data["summary_nodes"]
        ]
        knowledge_node = KnowledgeNode(summary_nodes=summary_nodes)
        knowledge_node.model = data.get("model", "gpt-3.5-turbo")
        knowledge_node.id = uuid.UUID(data["id"])
        knowledge_node.content = data["content"]
        knowledge_node.embedding = data["embedding"]
        knowledge_node.topic = data["topic"]
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

    def __init__(self, model="gpt-3.5-turbo"):
        self.logs: list = []
        self.summary_nodes: list = []
        self.knowledge_nodes: list = []
        self.rolling_window_size = 20
        self.model = model

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
        system = "You are a decision module that determines if information relevant to a topic is present in an article"

        prompt = (
            f"Given the following summary:\n\n{summary_node.content}\n\n"
            f"and the following knowledge base article:\n\n{knowledge_node.content}\n\n"
            "Please classify whether the summary has relevant information that can be added to the knowledge base article.\n\n"
            "If the summary is not related or relevant to the knowledge base article, please answer with `<no>`\n\n"
            "If the summary is relevant to the knowledge base article, please answer with `<yes>`\n\n"
        )

        return Prompt(system=system, prompt=prompt, model=self.model)

    def _semantic_similarity(self, summary_node, n_nearest=1):
        """
        This method is responsible for calculating the semantic similarity between a summary node and the knowledge nodes.
        The method returns the n_nearest knowledge nodes to the summary node in embedding space
        """
        if len(self.knowledge_nodes) == 0:
            return [None]

        embedding = summary_node.embedding

        similarities = []

        for knowledge_node in self.knowledge_nodes:
            similarity = cosine_similarity(embedding, knowledge_node.embedding)
            similarities.append([similarity, knowledge_node])

        # Sort the similarities list in descending order based on similarity value
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Take the top n_nearest knowledge nodes
        most_similar = similarities[:n_nearest]

        if len(most_similar) == 0:
            return [None]

        found_nodes = []
        for similarity, knowledge_node in most_similar:
            if "<yes>" in self._llm_classification(summary_node, knowledge_node):
                found_nodes.append(knowledge_node)

        if len(found_nodes) == 0:
            return [None]

        return found_nodes

    def build_summary_node(self, n_nearest=3) -> None:
        """After a rolling window of X logs, we build a summary node that summarizes the logs"""
        summary_node = SummaryNode(self.logs, model=self.model)
        summary_node.generate_summary()
        logging.info("<created summary node>")
        self.summary_nodes.append(summary_node)
        self.logs = []

        # If there are no knowledge nodes, we create one with the summary node
        # If there are knowledge nodes, we check if the summary node is similar to any of them
        # If it is, we update the closest knowledge base article with the summary node
        similar_knowledge_nodes = self._semantic_similarity(summary_node, n_nearest)
        existing_topics = []
        for node in similar_knowledge_nodes:
            if node is not None:
                node.summary_nodes.append(summary_node)
                node.update_article(summary_node, node.topic)
                logging.info(f"<updated knowledge node: {node.topic}>")
                logging.info(f"<> {node.content} <>")
                existing_topics.append(node.topic)

        new_topics = self.create_new_topics(summary_node.content, existing_topics)
        logging.info(f"<> New topics found: {new_topics} <>")
        for topic in new_topics:
            logging.info(f"<creating new knowledge node about {topic}>")
            new_node = KnowledgeNode(summary_nodes=[summary_node], model=self.model)
            new_node.generate_article(topic)
            new_node.topic = topic
            self.knowledge_nodes.append(new_node)

    @chat_gpt_prompt
    def _new_topics_prompt(self, summary: str, existing_topics):
        existing_topics = [str(x) for x in existing_topics]
        topics_string = "[" + "; ".join(existing_topics) + "]"

        system = (
            "You are topicGPT, based on INFORMATION create a list of new topics, that covers the part of the INFORMATION, that "
            f"is not already covered by the EXISTING TOPICS. Here an example: "
            "INFORMATION: The user is John. John loves playing tennis. The next presidential election in the USA will happen next year."
            "The year is 2023 or shortly after. GPT-4 was released, a transformer in the nlp domain, that greatly advanced the field"
            "EXISTING TOPICS: [John; Artificial intelligence] new topics: [Presidential election (USA)]"
        )

        prompt = f"INFORMATION:  {summary}, EXISTING TOPICS: {topics_string}\n\n"

        prompt += (
            f"TASK: Based on this INFORMATION create a list of new topics, that covers the part of the INFORMATION, that "
            f"is not already covered by the EXISTING TOPICS. Use as few new topics as possible to cover all of the INFORMATION that "
            f"is not covered by the existing topics. The name of a topic should be as concise as possible and capture the essence of the information that should be described."
            f"Only add a topic, when meaningful information, regarding that topic is in the INFORMATION"
            f"Return only the names of the new topics separated by ';'. Structure your output like this: [name1; name2; name3]. "
            f"If there are no new topics that would complement the EXISTING TOPICS, just return: [no topic found]"
        )

        return Prompt(system=system, prompt=prompt, model=self.model)

    def create_new_topics(self, summary: str, existing_topics: list):
        new_topics_string = self._new_topics_prompt(summary, existing_topics)
        if "no topic found" in new_topics_string.lower():
            return None

        new_topics_string = new_topics_string.replace("'", "")
        new_topics_string.replace("[", "")
        new_topics_string.replace("]", "")

        new_topics = new_topics_string.split(";")

        return new_topics

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
