import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from llm.base import ToolDefinition
from storage import database, files
from tools.base import Tool, ToolResult

SUPPORTED_CHARTS = [
    "histogram", "scatter", "boxplot",
    "correlation_heatmap", "countplot", "bar",
]


def _histogram(df, x, hue, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    if hue and hue in df.columns:
        for label, group in df.groupby(hue):
            ax.hist(group[x].dropna(), alpha=0.6, label=str(label), bins=30)
        ax.legend(title=hue)
    else:
        ax.hist(df[x].dropna(), bins=30, color="steelblue", edgecolor="white")
    ax.set_xlabel(x)
    ax.set_ylabel("Count")
    ax.set_title(title or f"Distribution of {x}")
    return fig


def _scatter(df, x, y, hue, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    if hue and hue in df.columns:
        for label, group in df.groupby(hue):
            ax.scatter(group[x], group[y], alpha=0.5, label=str(label), s=20)
        ax.legend(title=hue)
    else:
        ax.scatter(df[x], df[y], alpha=0.5, s=20, color="steelblue")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f"{x} vs {y}")
    return fig


def _boxplot(df, x, y, title):
    fig, ax = plt.subplots(figsize=(9, 6))
    order = df[x].value_counts().index.tolist()
    sns.boxplot(data=df, x=x, y=y, order=order, ax=ax, palette="Set2")
    ax.set_title(title or f"{y} by {x}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig


def _correlation_heatmap(df, title):
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(max(6, len(corr)), max(5, len(corr) - 1)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax)
    ax.set_title(title or "Correlation Heatmap")
    plt.tight_layout()
    return fig


def _countplot(df, x, hue, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    order = df[x].value_counts().index.tolist()
    sns.countplot(data=df, x=x, hue=hue if hue else None, order=order,
                  palette="Set2", ax=ax)
    ax.set_title(title or f"Count of {x}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig


def _bar(df, x, y, title):
    means = df.groupby(x)[y].mean().reset_index().sort_values(y, ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(means[x].astype(str), means[y], color="steelblue", edgecolor="white")
    ax.set_xlabel(x)
    ax.set_ylabel(f"Mean {y}")
    ax.set_title(title or f"Mean {y} by {x}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig


class CreateVisualizationTool(Tool):
    name = "create_visualization"
    description = (
        "Generate and save a visualization as an artifact. "
        f"Supported chart types: {', '.join(SUPPORTED_CHARTS)}. "
        "Returns the artifact_id for future reference."
    )

    def __init__(self, session_id: str):
        self.session_id = session_id

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset to visualise."},
                    "chart_type": {"type": "string", "enum": SUPPORTED_CHARTS},
                    "x_column": {"type": "string", "description": "Column for x-axis (or primary column)."},
                    "y_column": {"type": "string", "description": "Column for y-axis (scatter, boxplot, bar)."},
                    "hue_column": {"type": "string", "description": "Column to colour by (optional)."},
                    "title": {"type": "string", "description": "Chart title."},
                    "analysis_id": {"type": "string", "description": "Link this plot to an analysis record."},
                    "description": {"type": "string", "description": "Short description of what this plot shows."},
                },
                "required": ["dataset_id", "chart_type"],
            },
        )

    def execute(self, **kwargs) -> ToolResult:
        dataset_id = kwargs.get("dataset_id")
        chart_type = kwargs.get("chart_type", "").lower()
        x = kwargs.get("x_column")
        y = kwargs.get("y_column")
        hue = kwargs.get("hue_column")
        title = kwargs.get("title", "")
        analysis_id = kwargs.get("analysis_id")
        description = kwargs.get("description", "")

        if chart_type not in SUPPORTED_CHARTS:
            return ToolResult(status="error", error_type="ValueError",
                              message=f"chart_type must be one of: {SUPPORTED_CHARTS}")

        record = database.get_dataset(dataset_id)
        if not record:
            return ToolResult(status="error", error_type="NotFound",
                              message=f"Dataset '{dataset_id}' not found.")

        try:
            df = pd.read_csv(record["file_path"])
        except Exception as e:
            return ToolResult(status="error", error_type=type(e).__name__, message=str(e))

        # Validate required columns
        for col, label in [(x, "x_column"), (y, "y_column"), (hue, "hue_column")]:
            if col and col not in df.columns:
                available = ", ".join(df.columns.tolist())
                return ToolResult(status="error", error_type="ColumnNotFound",
                                  message=f"Column '{col}' not found.",
                                  details=f"Available columns: {available}")

        try:
            if chart_type == "histogram":
                if not x:
                    return ToolResult(status="error", error_type="ValueError",
                                      message="x_column is required for histogram.")
                fig = _histogram(df, x, hue, title)
                filename = f"histogram_{x}.png"

            elif chart_type == "scatter":
                if not x or not y:
                    return ToolResult(status="error", error_type="ValueError",
                                      message="x_column and y_column are required for scatter.")
                fig = _scatter(df, x, y, hue, title)
                filename = f"scatter_{x}_{y}.png"

            elif chart_type == "boxplot":
                if not x or not y:
                    return ToolResult(status="error", error_type="ValueError",
                                      message="x_column and y_column are required for boxplot.")
                fig = _boxplot(df, x, y, title)
                filename = f"boxplot_{y}_by_{x}.png"

            elif chart_type == "correlation_heatmap":
                fig = _correlation_heatmap(df, title)
                filename = "correlation_heatmap.png"

            elif chart_type == "countplot":
                if not x:
                    return ToolResult(status="error", error_type="ValueError",
                                      message="x_column is required for countplot.")
                fig = _countplot(df, x, hue, title)
                filename = f"countplot_{x}.png"

            elif chart_type == "bar":
                if not x or not y:
                    return ToolResult(status="error", error_type="ValueError",
                                      message="x_column and y_column are required for bar chart.")
                fig = _bar(df, x, y, title)
                filename = f"bar_{y}_by_{x}.png"

        except Exception as e:
            return ToolResult(status="error", error_type=type(e).__name__, message=str(e))

        # Compute path first, then register and save
        import uuid
        artifact_id = f"art_{uuid.uuid4().hex[:8]}"
        save_path = files.artifact_path(self.session_id, artifact_id, filename)

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        database.register_artifact(
            session_id=self.session_id,
            dataset_id=dataset_id,
            analysis_id=analysis_id,
            artifact_type="plot",
            filename=filename,
            file_path=str(save_path),
            description=description or title,
        )

        return ToolResult(
            status="success",
            data={
                "artifact_id": artifact_id,
                "filename": filename,
                "path": str(save_path),
                "description": description or title,
            },
        )
