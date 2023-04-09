import logging
import os
import openai
import random
import time
from functools import wraps
from dataclasses import dataclass, field
from typing import Optional, List
from emergent.utils import num_tokens_from_messages


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError, openai.error.APIError),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                logging.error(f"Error: {e}")
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def openai_chat_completion(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


@retry_with_exponential_backoff
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


@dataclass
class Prompt:
    prompt: Optional[str] = None
    system: Optional[str] = None
    messages: Optional[List] = None
    model: str = "gpt-3.5-turbo"
    kwargs: dict = field(default_factory=dict)


def create_message_list(prompt: Prompt):
    messages = []

    if prompt.system:
        messages.append({"role": "system", "content": prompt.system})
    if prompt.messages:
        messages.extend(prompt.messages)
    if prompt.prompt:
        messages.append({"role": "user", "content": prompt.prompt})

    return messages


def chat_gpt_prompt(func):
    """A decorator that takes a function that creates a prompt and executes the result for a specific gpt model with a specific system prompt"""
    # Wondering if there is a better name for this

    @wraps(func)
    def wrapper(*args, **kwargs):
        prompt = func(*args, **kwargs)

        model = "gpt-3.5-turbo"
        kwargs = {
            "temperature": 0.5,
        }

        if isinstance(prompt, Prompt):
            messages = create_message_list(prompt)
            model = prompt.model
            kwargs.update(prompt.kwargs)
        elif isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            raise ValueError(
                "Returned value must be a string or emergent.Prompt object"
            )

        # Handles breach of GPT-3.5 token limit
        total_tokens = num_tokens_from_messages(messages, model=model)
        if total_tokens > 4096:
            logging.warning(
                "The number of tokens in the prompt exceeds the limit of of GPT-3.5 (4096 tokens). Temporarily switching to GPT-4."
            )
            model = "gpt-4"

        response = openai_chat_completion(
            model=model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content

    return wrapper


def chat_gpt_kshot(func):
    """
    A decorator that takes a function that returns a list of messages
    It then sends that over to the chat completion api
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        messages = func(*args, **kwargs)
        response = openai_chat_completion(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.5,
        )
        return response.choices[0].message.content

    return wrapper
