import json
import time

from agent.tracer import TraceService
from context.builder import ContextBuilder
from llm.base import LLMProvider, ToolCall
from storage import database
from tools.base import Tool, ToolResult

MAX_ITERATIONS = 10
MAX_CONSECUTIVE_ERRORS = 3


def _result_summary(result: ToolResult) -> str:
    if result.status == "error":
        return f"error: {result.error_type}: {result.message}"
    d = result.data or {}
    if "summary" in d:
        return d["summary"]
    if "metrics" in d:
        return "metrics: " + ", ".join(f"{k}={v}" for k, v in list(d["metrics"].items())[:3])
    if "row_count" in d:
        return f"{d['row_count']} rows, {d.get('column_count', '?')} cols"
    if "groups" in d:
        return f"{len(d['groups'])} groups"
    if "stdout" in d:
        return "stdout: " + d["stdout"][:120]
    return "success"


def _extract_artifact_ids(result: ToolResult) -> list[str]:
    d = result.data or {}
    art = d.get("artifact_id") or d.get("artifact_ids", [])
    if isinstance(art, str):
        return [art]
    if isinstance(art, list):
        return [a for a in art if isinstance(a, str)]
    return []


class AgentRunner:
    def __init__(self, tools: list[Tool], context_builder: ContextBuilder, provider: LLMProvider):
        self.tools = {t.name: t for t in tools}
        self.context_builder = context_builder
        self.provider = provider

    def run(self, session_id: str, user_message: str) -> str:
        session_state = self._load_session_state(session_id)
        tracer = TraceService(session_id, user_message)

        user_msg = {"role": "user", "content": user_message}
        session_state["conversation_history"].append(user_msg)
        database.save_conversation_turn(session_id, user_msg)

        tool_definitions = [t.get_definition() for t in self.tools.values()]
        consecutive_errors = 0

        for _ in range(MAX_ITERATIONS):
            system = self.context_builder.build_system(session_state)
            messages = self.context_builder.build_messages(session_state)

            t0 = time.time()
            response = self.provider.complete(
                system=system,
                messages=messages,
                tools=tool_definitions,
            )
            llm_ms = (time.time() - t0) * 1000

            if response.stop_reason == "end_turn":
                assistant_msg = {"role": "assistant", "content": response.text}
                session_state["conversation_history"].append(assistant_msg)
                database.save_conversation_turn(session_id, assistant_msg)
                tracer.record_final_response(response.text, llm_ms)
                return response.text

            if response.stop_reason == "tool_use":
                assistant_msg = self.provider.format_assistant_tool_calls(response)
                session_state["conversation_history"].append(assistant_msg)
                database.save_conversation_turn(session_id, assistant_msg)

                results: list[tuple[ToolCall, ToolResult]] = []
                for tc in response.tool_calls:
                    t0 = time.time()
                    result = self._execute_tool(tc.name, tc.input)
                    tool_ms = (time.time() - t0) * 1000
                    results.append((tc, result))
                    tracer.record_tool_call(
                        tool_name=tc.name,
                        tool_args=tc.input,
                        result_status=result.status,
                        result_summary=_result_summary(result),
                        elapsed_ms=tool_ms,
                        artifact_ids=_extract_artifact_ids(result),
                    )

                serialized = [(tc, json.dumps(r.to_dict())) for tc, r in results]
                tool_msgs = self.provider.format_tool_results(serialized)
                for msg in tool_msgs:
                    session_state["conversation_history"].append(msg)
                    database.save_conversation_turn(session_id, msg)

                if all(r.status == "error" for _, r in results):
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    return (
                        "I was unable to complete this task — "
                        f"{MAX_CONSECUTIVE_ERRORS} consecutive tool errors occurred."
                    )
                continue

            break

        return "I was unable to complete this task within the allowed number of steps."

    def _execute_tool(self, tool_name: str, tool_input: dict) -> ToolResult:
        tool = self.tools.get(tool_name)
        if not tool:
            return ToolResult(
                status="error",
                error_type="UnknownTool",
                message=f"Tool '{tool_name}' is not registered.",
            )
        try:
            return tool.execute(**tool_input)
        except Exception as e:
            return ToolResult(
                status="error",
                error_type=type(e).__name__,
                message=str(e),
            )

    def _load_session_state(self, session_id: str) -> dict:
        return {
            "session_id": session_id,
            "conversation_history": database.get_conversation_history(session_id),
            "datasets":  database.get_session_datasets(session_id),
            "analyses":  database.get_session_analyses(session_id),
            "artifacts": database.get_session_artifacts(session_id),
        }
