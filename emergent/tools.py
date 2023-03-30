from dataclasses import dataclass, field
import functools
import inspect
import json

example_messages = [
    {
        "role": "user",
        "content": "[Example question from the user that might require memory access]",
        "name": "example_user",
    },
    {
        "role": "assistant",
        "content": 'search_memory({"query": "foobar"})\n-> [results from the tool]',
        "name": "example_assistant",
    },
]


def tool(
    name: str = None,
    desc: str = None,
    params: dict = None,
):
    """A decorator for creating tools.

    The tool description is the docstring of the function.
    The tool parameters are the function arguments, and the
    parameter descriptions are the annotations.
    """

    def decorator(func):
        tool_name = name
        tool_desc = desc
        tool_params = params

        if tool_name is None:
            tool_name = func.__name__
        if tool_desc is None:
            tool_desc = func.__doc__

        func.schema = Tool(
            name=tool_name,
            description=tool_desc,
            parameters=tool_params,
            function=func,
        )

        return func

    return decorator


@dataclass
class Tool:
    name: str
    description: str
    function: callable
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        self.signature = inspect.signature(self.function)

        # Populate tool_params if not provided
        if not self.parameters:
            params = list(self.signature.parameters.items())

            if self.is_method():
                params = params[1:]

            self.parameters = {
                k: f"The {k}" if v.annotation is inspect._empty else v.annotation
                for k, v in params
            }

    @property
    def usage(self):
        text = f"{self.name}({json.dumps(self.parameters)})"
        return text

    def is_method(self):
        first_param = list(self.signature.parameters.values())[0]
        return first_param.name == "self" and isinstance(first_param.annotation, type)
