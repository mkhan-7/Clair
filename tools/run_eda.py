import pandas as pd

from llm.base import ToolDefinition
from storage import database
from tools.base import Tool, ToolResult


def _build_summary(dataset_id: str, df: pd.DataFrame, notable_corrs: list[dict]) -> str:
    rows, cols = df.shape
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(include="object").columns.tolist()
    missing_cols = [c for c in df.columns if df[c].isna().any()]

    parts = [
        f"EDA on {dataset_id}: {rows} rows, {cols} columns.",
        f"Numeric columns: {', '.join(numeric) if numeric else 'none'}.",
        f"Categorical columns: {', '.join(categorical) if categorical else 'none'}.",
    ]
    if missing_cols:
        parts.append(f"Missing values in: {', '.join(missing_cols)}.")
    else:
        parts.append("No missing values.")
    if notable_corrs:
        corr_strs = [f"{c['col_a']} & {c['col_b']} (r={c['r']})" for c in notable_corrs[:3]]
        parts.append(f"Notable correlations: {', '.join(corr_strs)}.")
    return " ".join(parts)


class RunEdaTool(Tool):
    name = "run_eda"
    description = (
        "Run exploratory data analysis on a dataset. Returns descriptive statistics, "
        "missingness analysis, distribution summaries, correlation analysis, and "
        "optional target-variable breakdown. Records the analysis for future reference."
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
                    "dataset_id": {
                        "type": "string",
                        "description": "Dataset to analyse.",
                    },
                    "target_column": {
                        "type": "string",
                        "description": "Optional target variable for supervised-learning context.",
                    },
                },
                "required": ["dataset_id"],
            },
        )

    def execute(self, **kwargs) -> ToolResult:
        dataset_id = kwargs.get("dataset_id")
        target_column = kwargs.get("target_column")

        record = database.get_dataset(dataset_id)
        if not record:
            return ToolResult(status="error", error_type="NotFound",
                              message=f"Dataset '{dataset_id}' not found.")

        try:
            df = pd.read_csv(record["file_path"])
        except Exception as e:
            return ToolResult(status="error", error_type=type(e).__name__, message=str(e))

        rows, cols = df.shape

        # Per-column analysis
        columns = []
        for col in df.columns:
            missing = int(df[col].isna().sum())
            info: dict = {
                "name": col,
                "dtype": str(df[col].dtype),
                "missing_count": missing,
                "missing_pct": round(missing / rows * 100, 1) if rows else 0.0,
            }
            if df[col].dtype == object:
                vc = df[col].value_counts()
                info["cardinality"] = int(df[col].nunique())
                info["top_values"] = {str(k): int(v) for k, v in vc.head(5).items()}
            else:
                s = df[col].describe()
                info["stats"] = {k: round(float(v), 4) for k, v in s.items()}
                info["skewness"] = round(float(df[col].skew()), 4)
                info["kurtosis"] = round(float(df[col].kurt()), 4)
            columns.append(info)

        # Correlations (numeric only)
        numeric_df = df.select_dtypes(include="number")
        notable_corrs: list[dict] = []
        corr_matrix: dict = {}
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr().round(4)
            corr_matrix = corr.to_dict()
            seen: set = set()
            for a in corr.columns:
                for b in corr.columns:
                    if a >= b:
                        continue
                    r = float(corr.loc[a, b])
                    if abs(r) >= 0.3:
                        key = tuple(sorted([a, b]))
                        if key not in seen:
                            seen.add(key)
                            notable_corrs.append({"col_a": a, "col_b": b, "r": round(r, 4)})
            notable_corrs.sort(key=lambda x: abs(x["r"]), reverse=True)

        # Target breakdown
        target_analysis = None
        if target_column and target_column in df.columns:
            try:
                if df[target_column].dtype == object:
                    target_analysis = {"value_counts": df[target_column].value_counts().to_dict()}
                else:
                    groups = {}
                    for col in numeric_df.columns:
                        if col == target_column:
                            continue
                        groups[col] = round(float(df[col].corr(df[target_column])), 4)
                    target_analysis = {"correlations_with_target": groups}
            except Exception:
                pass

        summary = _build_summary(dataset_id, df, notable_corrs)
        analysis_id = database.create_analysis(
            session_id=self.session_id,
            dataset_id=dataset_id,
            analysis_type="EDA",
            summary=summary,
        )

        return ToolResult(
            status="success",
            data={
                "analysis_id": analysis_id,
                "dataset_id": dataset_id,
                "shape": {"rows": rows, "columns": cols},
                "columns": columns,
                "notable_correlations": notable_corrs,
                "target_analysis": target_analysis,
                "summary": summary,
            },
        )
