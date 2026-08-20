import json
import uuid

from storage import database


class TraceService:
    """Records one agent run (one user message → final response) into the traces table."""

    def __init__(self, session_id: str, user_message: str):
        self.session_id = session_id
        self.run_id = f"run_{uuid.uuid4().hex[:8]}"
        self.user_message = user_message
        self._step = 0

    def record_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
        result_status: str,
        result_summary: str,
        elapsed_ms: float,
        artifact_ids: list[str] | None = None,
    ) -> None:
        database.save_trace_step(
            run_id=self.run_id,
            session_id=self.session_id,
            step_index=self._step,
            user_message=self.user_message if self._step == 0 else None,
            llm_decision="tool_use",
            tool_name=tool_name,
            tool_args=json.dumps(tool_args),
            tool_result_status=result_status,
            tool_result_summary=result_summary,
            execution_time_ms=elapsed_ms,
            artifact_ids=json.dumps(artifact_ids or []),
            final_response=None,
        )
        self._step += 1

    def record_final_response(self, response: str, elapsed_ms: float) -> None:
        database.save_trace_step(
            run_id=self.run_id,
            session_id=self.session_id,
            step_index=self._step,
            user_message=self.user_message if self._step == 0 else None,
            llm_decision="end_turn",
            tool_name=None,
            tool_args=None,
            tool_result_status=None,
            tool_result_summary=None,
            execution_time_ms=elapsed_ms,
            artifact_ids=json.dumps([]),
            final_response=response[:600],
        )
        self._step += 1
