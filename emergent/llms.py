import logging
import os
import openai
import random
import time
from functools import wraps

from dataclasses import dataclass, field
from typing import Optional, List
from tiktoken

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

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

        # Calculate total tokens
        total_tokens = num_tokens_from_messages(messages, model=model)
        
        # Check if total tokens for request exceeds token limit
        if total_tokens > 4096:
            logging.warning("The number of tokens in the prompt exceeds the limit (4096 tokens). Skipping this prompt.")
            return
            
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
