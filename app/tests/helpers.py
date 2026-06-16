"""
tests/helpers.py — Test doubles for the LLM client.

These mimic just enough of the OpenAI SDK response shape that the agent loop
relies on (`response.choices[0].message` with `.content` and `.tool_calls`,
where each tool call has `.id` and `.function.name` / `.function.arguments`).
"""
from types import SimpleNamespace


def _tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def text_completion(content: str) -> SimpleNamespace:
    """A final assistant message with no tool calls."""
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def tool_call_completion(name: str, arguments: str, call_id: str = "call_1") -> SimpleNamespace:
    """An assistant message that requests a single tool call."""
    message = SimpleNamespace(
        content="",
        tool_calls=[_tool_call(call_id, name, arguments)],
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeLLM:
    """Stand-in for the OpenAI client.

    Returns the queued completions in order on each
    ``chat.completions.create(...)`` call. Records calls for assertions.
    """

    def __init__(self, completions):
        self._completions = list(completions)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *args, **kwargs):
        self.calls.append(kwargs)
        if not self._completions:
            raise AssertionError("FakeLLM ran out of queued completions")
        return self._completions.pop(0)
