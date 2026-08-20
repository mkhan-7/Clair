import numpy as np
import pandas as pd

from llm.base import ToolDefinition
from storage import database
from tools.base import Tool, ToolResult


class ProfileColumnTool(Tool):
    name = "profile_column"
    description = (
        "Deep profile of a single column: distribution, percentiles, outliers, "
        "and cardinality for categorical columns. Use when you need more detail "
        "than inspect_dataset provides for a specific variable."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset to profile."},
                    "column_name": {"type": "string", "description": "Column to profile."},
                },
                "required": ["dataset_id", "column_name"],
            },
        )

    def execute(self, **kwargs) -> ToolResult:
        dataset_id  = kwargs.get("dataset_id")
        column_name = kwargs.get("column_name")

        record = database.get_dataset(dataset_id)
        if not record:
            return ToolResult(status="error", error_type="NotFound",
                              message=f"Dataset '{dataset_id}' not found.")

        try:
            df = pd.read_csv(record["file_path"])
        except Exception as e:
            return ToolResult(status="error", error_type=type(e).__name__, message=str(e))

        if column_name not in df.columns:
            return ToolResult(status="error", error_type="ColumnNotFound",
                              message=f"Column '{column_name}' not found.",
                              details=f"Available: {', '.join(df.columns)}")

        col = df[column_name]
        n = len(col)
        missing = int(col.isna().sum())

        profile: dict = {
            "column": column_name,
            "dtype": str(col.dtype),
            "total_rows": n,
            "missing_count": missing,
            "missing_pct": round(missing / n * 100, 2) if n else 0.0,
        }

        if col.dtype == object:
            vc = col.value_counts(dropna=True)
            profile.update({
                "type": "categorical",
                "cardinality": int(col.nunique()),
                "mode": str(col.mode().iloc[0]) if not col.mode().empty else None,
                "top_10_values": {str(k): int(v) for k, v in vc.head(10).items()},
                "bottom_5_values": {str(k): int(v) for k, v in vc.tail(5).items()},
            })
        else:
            clean = col.dropna()
            q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
            iqr = q3 - q1
            outliers = int(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum())
            profile.update({
                "type": "numeric",
                "mean":   round(float(clean.mean()), 4),
                "median": round(float(clean.median()), 4),
                "std":    round(float(clean.std()), 4),
                "min":    round(float(clean.min()), 4),
                "max":    round(float(clean.max()), 4),
                "percentiles": {
                    "p5":  round(float(clean.quantile(0.05)), 4),
                    "p25": round(q1, 4),
                    "p50": round(float(clean.quantile(0.50)), 4),
                    "p75": round(q3, 4),
                    "p90": round(float(clean.quantile(0.90)), 4),
                    "p95": round(float(clean.quantile(0.95)), 4),
                    "p99": round(float(clean.quantile(0.99)), 4),
                },
                "skewness": round(float(clean.skew()), 4),
                "kurtosis": round(float(clean.kurt()), 4),
                "outlier_count_iqr": outliers,
                "outlier_pct_iqr":   round(outliers / len(clean) * 100, 2) if clean.size else 0.0,
            })

        return ToolResult(status="success", data=profile)
