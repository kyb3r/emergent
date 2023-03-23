import logging
import os
import openai
import random
import time
from functools import wraps


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


def chat_gpt_prompt(func):
    """A decorator that takes a function that creates a prompt and executes the result for a specific gpt model with a specific system prompt"""
    # Wondering if there is a better name for this

    @wraps(func)
    def wrapper(*args, **kwargs):
        system, prompt, model = func(*args, **kwargs)
        response = openai_chat_completion(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return response.choices[0].message.content

    return wrapper
