from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ExecutionResult:
    status: str  # "success" | "error" | "timeout"
    stdout: str = ""
    stderr: str = ""
    error_type: str | None = None
    error_message: str | None = None
    execution_time_ms: float = 0.0
    artifact_paths: list[str] = field(default_factory=list)


class ExecutionService(ABC):
    """Abstract interface for running user-supplied Python code.

    A local implementation is added in Slice 2. This abstraction lets a
    stronger sandbox replace it later without touching the agent or tools.
    """

    @abstractmethod
    def execute(
        self,
        code: str,
        working_dir: str,
        timeout_seconds: int = 30,
    ) -> ExecutionResult:
        ...
