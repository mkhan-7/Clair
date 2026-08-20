# Clair — LLM Data Scientist

A local, single-agent system that lets you ask natural-language questions about tabular datasets. Upload a CSV, ask analytical questions, and the agent inspects the data, selects tools, runs code, trains models, and explains results — autonomously.

Built as a Phase 1 exploration of **harness engineering for LLM agents**: the surrounding infrastructure (tool abstraction, context management, artifact tracking, error recovery, execution tracing) that makes a language model reliable for multi-step analytical work.

---

## Prerequisites

- Python 3.12
- An API key from [Anthropic](https://console.anthropic.com) (default) or [OpenAI](https://platform.openai.com)
- macOS or Linux (the local execution service uses subprocess isolation)

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url> ### update
cd Clair
```

### 2. Create and activate a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **macOS note:** XGBoost requires OpenMP. If you hit a library error on first run:
> ```bash
> brew install libomp
> ```

### 4. Configure your API key

Copy `.env` and fill in your key:

```bash
# .env is already present — open it and replace the placeholder:
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

To use OpenAI instead:

```bash
OPENAI_API_KEY=your_openai_api_key_here
LLM_PROVIDER=openai
```

---

## Running Clair

### Option A — Streamlit web UI (recommended)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. From there:

1. Click **+ New Session** in the sidebar
2. Upload a CSV file under **Datasets**
3. Type a question in the chat box

### Option B — CLI

```bash
# Create a session
python main.py new-session
# → sess_abc123...

# Upload a dataset
python main.py upload sess_abc123 path/to/data.csv
# → dataset_id: ds_xyz...

# Ask a question
python main.py chat sess_abc123 "What columns does this dataset have?"

# View the execution trace for a session
python main.py trace sess_abc123

# View full conversation history
python main.py history sess_abc123
```

---

## What the agent can do

| Capability | How to ask |
|---|---|
| Dataset overview | "What columns are there? Any quality issues?" |
| Column profile | "Show me the distribution of column X" |
| Exploratory analysis | "Run a full EDA on this dataset" |
| Visualizations | "Plot the correlation heatmap" / "Show a histogram of Y" |
| Group comparison | "Compare the mean of X across groups in Y" |
| Ad-hoc analysis | "Calculate Z for rows where A > 0.5" |
| Predictive modeling | "Build a model to predict D" |
| Model evaluation | "Cross-validate the model and compare accuracy vs F1" |
| Feature importance | "What are the top predictors?" |
| SHAP explainability | "Explain the model using SHAP" |
| Artifact retrieval | "Show me the confusion matrix again" |

---

## Project structure

```
Clair/
├── app.py                  # Streamlit frontend
├── main.py                 # CLI entry point
├── requirements.txt
├── .env                    # API keys (not committed)
│
├── agent/
│   ├── runner.py           # Main agent loop (bounded, tool-dispatching)
│   └── tracer.py           # Per-step execution tracing to SQLite
│
├── context/
│   └── builder.py          # Selective context construction for each LLM call
│
├── tools/                  # 11 tools — all implement Tool ABC
│   ├── base.py             # Tool + ToolResult abstract classes
│   ├── inspect_dataset.py
│   ├── profile_column.py
│   ├── run_eda.py
│   ├── create_visualization.py
│   ├── execute_python.py   # Subprocess execution + artifact registration
│   ├── run_model.py        # RF, Logistic, XGBoost training
│   ├── recommend_model.py
│   ├── evaluate_model.py
│   ├── compare_groups.py
│   └── artifacts.py        # list_artifacts + get_artifact
│
├── executor/
│   ├── base.py             # ExecutionService ABC
│   └── local.py            # Subprocess-based implementation
│
├── llm/
│   ├── base.py             # LLMProvider ABC
│   ├── anthropic_provider.py
│   └── openai_provider.py
│
├── storage/
│   ├── database.py         # SQLite — sessions, datasets, artifacts, traces
│   └── files.py            # Filesystem — CSVs, plots, model pickles
│
├── sample_datasets/        # Example CSVs to get started
│
└── notes/                  # Project documentation
    ├── spec.md             # Original project specification
    ├── tracing_examples.txt # Annotated execution traces for three scenarios
    └── process.txt         # Development notes
```

**Runtime directories** (created automatically, gitignored):

```
data/
├── clair.db                # SQLite database
└── sessions/<session_id>/
    ├── datasets/           # Uploaded CSV files
    ├── artifacts/          # Generated plots, model files, tables
    └── workspace/          # Subprocess working directory
```

---

## Switching providers

Set `LLM_PROVIDER` in `.env`:

```bash
LLM_PROVIDER=anthropic   # uses claude-sonnet-4-5 (default)
LLM_PROVIDER=openai      # uses gpt-4o
```

To pin a specific model:

```bash
LLM_MODEL=claude-sonnet-4-5
LLM_MODEL=gpt-4o
```

---

## Architecture notes

- **Agent loop**: bounded at `MAX_ITERATIONS=10` tool calls per request and `MAX_CONSECUTIVE_ERRORS=3` before aborting
- **Context**: the `ContextBuilder` passes a sliding 20-turn window; older tool results are compressed to summaries
- **Artifacts**: every generated file (plot, model, table) is registered in SQLite and referenced by ID — the agent can re-display prior results without re-running analysis
- **Execution**: Python code runs in a subprocess with a 30-second timeout; any files it generates are automatically moved to the artifact store
- **Tracing**: every tool call is logged to the `traces` table — query with `python main.py trace <session_id>`
