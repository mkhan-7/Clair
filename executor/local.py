import subprocess
import sys
import time
from pathlib import Path

from executor.base import ExecutionResult, ExecutionService


def _parse_error(stderr: str) -> tuple[str, str]:
    """Extract error type and message from a Python traceback."""
    lines = [l for l in stderr.strip().splitlines() if l.strip()]
    if not lines:
        return "UnknownError", "An unknown error occurred."
    last = lines[-1]
    if ":" in last:
        etype, _, msg = last.partition(":")
        return etype.strip(), msg.strip()
    return "RuntimeError", last.strip()


class LocalExecutionService(ExecutionService):
    """Runs code in a subprocess using the current Python interpreter.

    Isolated to a per-session working directory. Designed to be replaced
    by a stronger sandbox without changing the tool or agent layer.
    """

    def __init__(self, python_executable: str = sys.executable):
        self.python = python_executable

    def execute(self, code: str, working_dir: str, timeout_seconds: int = 30) -> ExecutionResult:
        work = Path(working_dir).resolve()
        work.mkdir(parents=True, exist_ok=True)

        script = work / "_run.py"
        script.write_text(code, encoding="utf-8")

        before = {p for p in work.iterdir() if p != script}
        t0 = time.time()

        try:
            proc = subprocess.run(
                [self.python, str(script)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(work),
            )
            elapsed_ms = (time.time() - t0) * 1000
            after = {p for p in work.iterdir() if p != script}
            new_files = [str(p) for p in (after - before)]

            if proc.returncode == 0:
                return ExecutionResult(
                    status="success",
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    execution_time_ms=elapsed_ms,
                    artifact_paths=new_files,
                )

            error_type, error_msg = _parse_error(proc.stderr)
            return ExecutionResult(
                status="error",
                stdout=proc.stdout,
                stderr=proc.stderr,
                error_type=error_type,
                error_message=error_msg,
                execution_time_ms=elapsed_ms,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                status="timeout",
                error_type="TimeoutError",
                error_message=f"Execution exceeded {timeout_seconds}s limit.",
                execution_time_ms=timeout_seconds * 1000,
            )
        except Exception as e:
            return ExecutionResult(
                status="error",
                error_type=type(e).__name__,
                error_message=str(e),
            )
