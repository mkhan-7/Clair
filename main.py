#!/usr/bin/env python3
"""Clair CLI — LLM Data Scientist (Phase 1)

Usage:
    python main.py new-session
    python main.py upload <session_id> <path/to/file.csv>
    python main.py chat   <session_id> "<question>"
"""
import json
import os
import sys
import uuid
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from storage import database, files
from tools.inspect_dataset import InspectDatasetTool
from tools.execute_python import ExecutePythonTool
from tools.run_eda import RunEdaTool
from tools.create_visualization import CreateVisualizationTool
from tools.artifacts import ListArtifactsTool, GetArtifactTool
from tools.run_model import RunModelTool
from tools.recommend_model import RecommendModelTool
from tools.profile_column import ProfileColumnTool
from tools.compare_groups import CompareGroupsTool
from tools.evaluate_model import EvaluateModelTool
from executor.local import LocalExecutionService
from context.builder import ContextBuilder
from agent.runner import AgentRunner
from llm.base import LLMProvider


def _build_provider() -> LLMProvider:
    provider_name = os.getenv("LLM_PROVIDER", "anthropic").lower()
    model = os.getenv("LLM_MODEL")  # None → each provider uses its own default

    if provider_name == "openai":
        from llm.openai_provider import OpenAIProvider
        return OpenAIProvider(model=model or "gpt-4o")

    from llm.anthropic_provider import AnthropicProvider
    return AnthropicProvider(model=model or "claude-sonnet-4-5")


def cmd_new_session(_args):
    database.init_db()
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    database.create_session(session_id)
    print(session_id)


def cmd_upload(args):
    database.init_db()

    source = Path(args.csv)
    if not source.exists():
        print(f"Error: file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)
    if source.suffix.lower() != ".csv":
        print("Error: only CSV files are supported.", file=sys.stderr)
        sys.exit(1)

    session = database.get_session(args.session_id)
    if not session:
        print(f"Error: session '{args.session_id}' not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(source)
    row_count, col_count = df.shape

    dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
    dest_path = files.save_dataset(args.session_id, dataset_id, str(source))
    database.register_dataset(
        session_id=args.session_id,
        dataset_id=dataset_id,
        file_path=dest_path,
        original_filename=source.name,
        row_count=row_count,
        column_count=col_count,
    )
    print(f"dataset_id: {dataset_id}")
    print(f"rows: {row_count}  columns: {col_count}")


def cmd_chat(args):
    database.init_db()

    session = database.get_session(args.session_id)
    if not session:
        print(f"Error: session '{args.session_id}' not found.", file=sys.stderr)
        sys.exit(1)

    sid = args.session_id
    executor = LocalExecutionService()
    tools = [
        InspectDatasetTool(),
        ExecutePythonTool(execution_service=executor, session_id=sid),
        RunEdaTool(session_id=sid),
        CreateVisualizationTool(session_id=sid),
        ListArtifactsTool(session_id=sid),
        GetArtifactTool(),
        RecommendModelTool(),
        RunModelTool(session_id=sid),
        ProfileColumnTool(),
        CompareGroupsTool(),
        EvaluateModelTool(),
    ]
    runner = AgentRunner(tools=tools, context_builder=ContextBuilder(), provider=_build_provider())
    response = runner.run(session_id=args.session_id, user_message=args.message)
    print(response)


def cmd_trace(args):
    database.init_db()
    steps = database.get_session_traces(args.session_id)
    if not steps:
        print(f"No trace found for session '{args.session_id}'.")
        return

    print(f"\n=== Execution Trace: {args.session_id} ===\n")

    # Group steps by run_id, preserving insertion order
    runs: dict[str, list[dict]] = {}
    for s in steps:
        runs.setdefault(s["run_id"], []).append(s)

    for run_id, run_steps in runs.items():
        # Print run header with timestamp from first step
        ts = run_steps[0]["created_at"][:19].replace("T", " ")
        user_msg = next((s["user_message"] for s in run_steps if s["user_message"]), "")
        print(f"Run  {run_id}  │  {ts}")
        if user_msg:
            snippet = user_msg[:120] + ("…" if len(user_msg) > 120 else "")
            print(f'  User: "{snippet}"')
        print("  " + "─" * 70)

        for s in run_steps:
            idx = s["step_index"]
            decision = s["llm_decision"]
            ms = s["execution_time_ms"] or 0.0

            if decision == "tool_use":
                name = s["tool_name"] or "?"
                status = s["tool_result_status"] or "?"
                summary = s["tool_result_summary"] or ""
                arts_raw = s["artifact_ids"] or "[]"
                try:
                    arts = json.loads(arts_raw)
                except Exception:
                    arts = []

                summary_snippet = summary[:80] + ("…" if len(summary) > 80 else "")
                art_str = f"  artifacts={arts}" if arts else ""
                print(f"  [{idx}] tool_use  {name:<24} {ms:6.0f}ms  → {status}: {summary_snippet}{art_str}")
            else:
                resp = s["final_response"] or ""
                resp_snippet = resp[:80] + ("…" if len(resp) > 80 else "")
                print(f'  [{idx}] end_turn  {"─":<24} {ms:6.0f}ms  → "{resp_snippet}"')

        print()


def cmd_history(args):
    database.init_db()
    history = database.get_conversation_history(args.session_id)
    if not history:
        print(f"No conversation history for session '{args.session_id}'.")
        return

    print(f"\n=== Conversation: {args.session_id} ===\n")
    for i, msg in enumerate(history, 1):
        role = msg.get("role", "?")
        content = msg.get("content", "")

        if isinstance(content, str):
            snippet = content[:120] + ("…" if len(content) > 120 else "")
            print(f"[{i:>3}] {role:<12}  {snippet!r}")
        elif isinstance(content, list):
            # Complex content: tool calls or tool results
            types = [b.get("type", "?") for b in content if isinstance(b, dict)]
            names = [b.get("name", "") for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
            if names:
                print(f"[{i:>3}] {role:<12}  [tool calls: {', '.join(names)}]")
            elif "tool_result" in types:
                print(f"[{i:>3}] {role:<12}  [{len(types)} tool result(s)]")
            else:
                print(f"[{i:>3}] {role:<12}  [{', '.join(types)}]")
        else:
            print(f"[{i:>3}] {role:<12}  (non-text content)")
    print()


def main():
    parser = argparse.ArgumentParser(prog="clair", description="LLM Data Scientist CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("new-session", help="Create a new session")

    p_upload = sub.add_parser("upload", help="Register a CSV dataset")
    p_upload.add_argument("session_id")
    p_upload.add_argument("csv", help="Path to CSV file")

    p_chat = sub.add_parser("chat", help="Ask a question in a session")
    p_chat.add_argument("session_id")
    p_chat.add_argument("message")

    p_trace = sub.add_parser("trace", help="Dump the execution trace for a session")
    p_trace.add_argument("session_id")

    p_history = sub.add_parser("history", help="Show conversation history for a session")
    p_history.add_argument("session_id")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "new-session": cmd_new_session,
        "upload": cmd_upload,
        "chat": cmd_chat,
        "trace": cmd_trace,
        "history": cmd_history,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
