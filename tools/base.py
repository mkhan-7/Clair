from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from llm.base import ToolDefinition


@dataclass
class ToolResult:
    status: str  # "success" | "error"
    data: Any = None
    error_type: str | None = None
    message: str | None = None
    details: str | None = None

    def to_dict(self) -> dict:
        d: dict = {"status": self.status}
        if self.data is not None:
            d["data"] = self.data
        if self.error_type:
            d["error_type"] = self.error_type
        if self.message:
            d["message"] = self.message
        if self.details:
            d["details"] = self.details
        return d


class Tool(ABC):
    name: str
    description: str

    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """Return provider-agnostic tool definition."""
        ...

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        ...
