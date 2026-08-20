from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolDefinition:
    """Provider-agnostic tool definition. Each provider translates this to its own wire format."""
    name: str
    description: str
    parameters: dict  # JSON Schema: {type: object, properties: {...}, required: [...]}


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    stop_reason: str          # "end_turn" | "tool_use"
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = field(default=None, repr=False)  # provider-internal; do not use outside providers


class LLMProvider(ABC):
    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[dict],
        tools: list[ToolDefinition],
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a request to the LLM and return a unified response."""
        ...

    @abstractmethod
    def format_assistant_tool_calls(self, response: LLMResponse) -> dict:
        """Produce the assistant message dict to append to history after a tool_use response."""
        ...

    @abstractmethod
    def format_tool_results(self, results: list[tuple[ToolCall, str]]) -> list[dict]:
        """Produce the tool result message(s) to append to history after executing tools."""
        ...
