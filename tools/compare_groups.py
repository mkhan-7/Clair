import pandas as pd
import numpy as np

from llm.base import ToolDefinition
from storage import database
from tools.base import Tool, ToolResult

DEFAULT_AGGS = ["mean", "median", "std", "count"]


class CompareGroupsTool(Tool):
    name = "compare_groups"
    description = (
        "Compare a numeric metric across groups defined by a categorical column. "
        "Returns per-group statistics (mean, median, std, count). "
        "Use for questions like 'recidivism rate by race' or 'average score by age group'."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "dataset_id":    {"type": "string", "description": "Dataset to analyse."},
                    "group_column":  {"type": "string", "description": "Categorical column defining the groups."},
                    "metric_column": {"type": "string", "description": "Numeric column to compare across groups."},
                    "aggregations":  {
                        "type": "array",
                        "items": {"type": "string", "enum": ["mean", "median", "std", "min", "max", "count", "sum"]},
                        "description": "Statistics to compute per group (default: mean, median, std, count).",
                    },
                    "filter_expr": {
                        "type": "string",
                        "description": "Optional pandas query string to filter rows before grouping, e.g. \"sex == 'Male'\".",
                    },
                },
                "required": ["dataset_id", "group_column", "metric_column"],
            },
        )

    def execute(self, **kwargs) -> ToolResult:
        dataset_id    = kwargs.get("dataset_id")
        group_column  = kwargs.get("group_column")
        metric_column = kwargs.get("metric_column")
        aggregations  = kwargs.get("aggregations") or DEFAULT_AGGS
        filter_expr   = kwargs.get("filter_expr")

        record = database.get_dataset(dataset_id)
        if not record:
            return ToolResult(status="error", error_type="NotFound",
                              message=f"Dataset '{dataset_id}' not found.")

        try:
            df = pd.read_csv(record["file_path"])
        except Exception as e:
            return ToolResult(status="error", error_type=type(e).__name__, message=str(e))

        for col, label in [(group_column, "group_column"), (metric_column, "metric_column")]:
            if col not in df.columns:
                return ToolResult(status="error", error_type="ColumnNotFound",
                                  message=f"{label} '{col}' not found.",
                                  details=f"Available: {', '.join(df.columns)}")

        if filter_expr:
            try:
                df = df.query(filter_expr)
            except Exception as e:
                return ToolResult(status="error", error_type="FilterError",
                                  message=f"filter_expr failed: {e}")

        if df[metric_column].dtype == object:
            return ToolResult(status="error", error_type="TypeError",
                              message=f"metric_column '{metric_column}' must be numeric.")

        valid_aggs = [a for a in aggregations if a in ["mean", "median", "std", "min", "max", "count", "sum"]]
        grouped = df.groupby(group_column)[metric_column].agg(valid_aggs).round(4)
        grouped = grouped.sort_values("mean" if "mean" in valid_aggs else valid_aggs[0], ascending=False)

        result_rows = {
            str(idx): {k: (int(v) if k == "count" else round(float(v), 4))
                       for k, v in row.items()}
            for idx, row in grouped.iterrows()
        }

        return ToolResult(
            status="success",
            data={
                "group_column":  group_column,
                "metric_column": metric_column,
                "aggregations":  valid_aggs,
                "filter":        filter_expr,
                "total_rows":    len(df),
                "groups":        result_rows,
            },
        )
