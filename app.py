#!/usr/bin/env python3
"""Clair — LLM Data Scientist  (Streamlit frontend)

Run from the project root:
    streamlit run app.py
"""
import os
import re
import sys
import uuid
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from storage import database, files

database.init_db()

from executor.local import LocalExecutionService
from context.builder import ContextBuilder
from agent.runner import AgentRunner
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


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Clair — LLM Data Scientist",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Remove default Streamlit top padding
st.markdown(
    "<style>div.block-container{padding-top:1.5rem}</style>",
    unsafe_allow_html=True,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_provider():
    provider_name = os.getenv("LLM_PROVIDER", "anthropic").lower()
    model = os.getenv("LLM_MODEL")
    if provider_name == "openai":
        from llm.openai_provider import OpenAIProvider
        return OpenAIProvider(model=model or "gpt-4o")
    from llm.anthropic_provider import AnthropicProvider
    return AnthropicProvider(model=model or "claude-sonnet-4-5")


def _make_runner(session_id: str) -> AgentRunner:
    executor = LocalExecutionService()
    tools = [
        InspectDatasetTool(),
        ExecutePythonTool(execution_service=executor, session_id=session_id),
        RunEdaTool(session_id=session_id),
        CreateVisualizationTool(session_id=session_id),
        ListArtifactsTool(session_id=session_id),
        GetArtifactTool(),
        RecommendModelTool(),
        RunModelTool(session_id=session_id),
        ProfileColumnTool(),
        CompareGroupsTool(),
        EvaluateModelTool(),
    ]
    return AgentRunner(
        tools=tools,
        context_builder=ContextBuilder(),
        provider=_build_provider(),
    )


def _load_display_history(session_id: str) -> list[dict]:
    """Extract only user/assistant text messages from persisted conversation history.
    Tool call and tool result messages (content=list) are filtered out for display."""
    raw = database.get_conversation_history(session_id)
    out = []
    for msg in raw:
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            out.append({"role": msg["role"], "content": content, "new_artifacts": []})
    return out


def _render_artifact(art: dict) -> None:
    """Render one artifact inline after a chat message."""
    p = art.get("file_path", "")
    if not p or not Path(p).exists():
        return
    suffix = Path(p).suffix.lower()
    caption = f"{art.get('description') or art['filename']}  ·  `{art['id']}`"
    if suffix in (".png", ".jpg", ".jpeg"):
        st.image(p, caption=caption, width=420)
    elif suffix == ".csv":
        try:
            df = pd.read_csv(p)
            st.caption(caption)
            st.dataframe(df.head(50), use_container_width=True)
        except Exception:
            st.caption(f"Artifact: {art['filename']}  (`{art['id']}`)")
    else:
        st.caption(f"📦 {art['filename']}  (`{art['id']}`)")


# ─── Session state ─────────────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = set()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 Clair")
    st.caption("Local LLM Data Scientist")
    st.divider()

    # ── Sessions
    st.markdown("**Session**")

    if st.button("＋ New Session", use_container_width=True):
        new_sid = f"sess_{uuid.uuid4().hex[:12]}"
        database.create_session(new_sid)
        st.session_state.session_id = new_sid
        st.session_state.display_messages = []
        st.session_state.uploaded_file_names = set()
        st.rerun()

    all_sessions = database.get_all_sessions()
    if all_sessions:
        options = [s["id"] for s in all_sessions]
        labels = {
            s["id"]: f"{s['id'][:18]}…  ({s['created_at'][:10]})"
            for s in all_sessions
        }
        current_idx = (
            options.index(st.session_state.session_id)
            if st.session_state.session_id in options
            else 0
        )
        chosen = st.selectbox(
            "Continue session",
            options,
            index=current_idx,
            format_func=lambda sid: labels[sid],
            label_visibility="collapsed",
        )
        if chosen != st.session_state.session_id:
            st.session_state.session_id = chosen
            st.session_state.display_messages = _load_display_history(chosen)
            st.session_state.uploaded_file_names = set()
            st.rerun()

    # ── Dataset upload (only when a session is active)
    if st.session_state.session_id:
        st.divider()
        st.markdown("**Datasets**")

        existing_ds = database.get_session_datasets(st.session_state.session_id)
        for ds in existing_ds:
            st.caption(f"✓ {ds['original_filename']}  ({ds['row_count']:,} rows × {ds['column_count']} cols)")

        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded and uploaded.name not in st.session_state.uploaded_file_names:
            with st.spinner("Registering dataset…"):
                tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
                try:
                    tmp.write(uploaded.getbuffer())
                    tmp.close()
                    df_up = pd.read_csv(tmp.name)
                    n_rows, n_cols = df_up.shape
                    ds_id = f"ds_{uuid.uuid4().hex[:8]}"
                    dest = files.save_dataset(st.session_state.session_id, ds_id, tmp.name)
                    database.register_dataset(
                        session_id=st.session_state.session_id,
                        dataset_id=ds_id,
                        file_path=dest,
                        original_filename=uploaded.name,
                        row_count=n_rows,
                        column_count=n_cols,
                    )
                    st.session_state.uploaded_file_names.add(uploaded.name)
                    st.success(f"Registered `{ds_id}`  ({n_rows:,} rows × {n_cols} cols)")
                except Exception as e:
                    st.error(f"Upload failed: {e}")
                finally:
                    Path(tmp.name).unlink(missing_ok=True)

        # ── Artifact browser
        artifacts = database.get_session_artifacts(st.session_state.session_id)
        if artifacts:
            st.divider()
            st.markdown("**Artifacts**")
            type_icons = {"plot": "🖼️", "table": "📊", "model": "🤖"}
            for art in artifacts:
                icon = type_icons.get(art["artifact_type"], "📦")
                with st.expander(f"{icon}  {art['filename']}"):
                    st.caption(f"`{art['id']}`")
                    if art.get("description"):
                        st.caption(art["description"])
                    p = art.get("file_path", "")
                    if p and Path(p).exists():
                        suffix = Path(p).suffix.lower()
                        if suffix in (".png", ".jpg", ".jpeg"):
                            st.image(p, use_container_width=True)
                        elif suffix == ".csv":
                            try:
                                df_art = pd.read_csv(p)
                                st.dataframe(df_art.head(20), use_container_width=True)
                            except Exception:
                                pass
                        with open(p, "rb") as fh:
                            st.download_button(
                                "⬇ Download",
                                fh,
                                file_name=art["filename"],
                                key=f"dl_{art['id']}",
                            )


# ─── Main area ────────────────────────────────────────────────────────────────

if not st.session_state.session_id:
    st.title("Clair — LLM Data Scientist")
    st.markdown(
        "Create a **new session** from the sidebar, upload a CSV dataset, "
        "and start asking analytical questions in plain language."
    )
    st.info("The agent can inspect datasets, run EDA, build predictive models, "
            "create visualizations, and explain results — all autonomously.")
    st.stop()

session_id = st.session_state.session_id
provider_label = os.getenv("LLM_PROVIDER", "anthropic").lower()
st.markdown(
    f"**Clair** &nbsp;<span style='color:gray;font-size:0.8em'>`{session_id}` · {provider_label}</span>",
    unsafe_allow_html=True,
)
st.divider()

# Render conversation history
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    for art in msg.get("new_artifacts", []):
        _render_artifact(art)

# Chat input
prompt = st.chat_input("Ask a question about your data…")
if prompt:
    datasets = database.get_session_datasets(session_id)
    if not datasets:
        st.warning("Upload a CSV dataset first (sidebar → Datasets).")
        st.stop()

    # Persist and immediately render user message
    st.session_state.display_messages.append(
        {"role": "user", "content": prompt, "new_artifacts": []}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Snapshot artifact IDs before the run
    before_ids = {a["id"] for a in database.get_session_artifacts(session_id)}

    # Run agent — render response inside assistant bubble
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                runner = _make_runner(session_id)
                response = runner.run(session_id=session_id, user_message=prompt)
            except Exception as e:
                response = f"An error occurred: {e}"
        st.markdown(response)

    # Collect new artifacts + any existing ones the agent referenced by ID
    all_arts = database.get_session_artifacts(session_id)
    art_map = {a["id"]: a for a in all_arts}
    new_arts = [a for a in all_arts if a["id"] not in before_ids]
    referenced_ids = set(re.findall(r'\bart_[a-f0-9]{8}\b', response))
    existing_referenced = [
        art_map[aid] for aid in referenced_ids
        if aid in art_map and aid in before_ids
        and art_map[aid]["artifact_type"] in ("plot", "table")
    ]
    render_arts = new_arts + existing_referenced
    for art in render_arts:
        _render_artifact(art)

    # Save assistant message (with artifact refs) for future renders
    st.session_state.display_messages.append(
        {"role": "assistant", "content": response, "new_artifacts": render_arts}
    )
