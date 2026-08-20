import pandas as pd

from llm.base import ToolDefinition
from storage import database
from tools.base import Tool, ToolResult


class InspectDatasetTool(Tool):
    name = "inspect_dataset"
    description = (
        "Inspect a registered dataset. Returns row/column counts, column names, "
        "data types, missing value counts, descriptive statistics for numeric columns, "
        "categorical cardinality, and a 5-row sample."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "The dataset identifier to inspect.",
                    }
                },
                "required": ["dataset_id"],
            },
        )

    def execute(self, **kwargs) -> ToolResult:
        dataset_id = kwargs.get("dataset_id")
        if not dataset_id:
            return ToolResult(
                status="error",
                error_type="ValueError",
                message="dataset_id is required.",
            )

        record = database.get_dataset(dataset_id)
        if not record:
            return ToolResult(
                status="error",
                error_type="NotFound",
                message=f"Dataset '{dataset_id}' not found.",
            )

        try:
            df = pd.read_csv(record["file_path"])
        except Exception as e:
            return ToolResult(
                status="error",
                error_type=type(e).__name__,
                message=str(e),
            )

        row_count, col_count = df.shape

        columns = []
        for col in df.columns:
            missing = int(df[col].isna().sum())
            info: dict = {
                "name": col,
                "dtype": str(df[col].dtype),
                "missing_count": missing,
                "missing_pct": round(missing / row_count * 100, 1) if row_count else 0.0,
            }
            if df[col].dtype == object:
                info["cardinality"] = int(df[col].nunique())
                info["top_values"] = df[col].value_counts().head(5).to_dict()
            columns.append(info)

        numeric = df.select_dtypes(include="number")
        stats = numeric.describe().round(4).to_dict() if not numeric.empty else {}

        sample = df.head(5).fillna("").to_dict(orient="records")

        return ToolResult(
            status="success",
            data={
                "dataset_id": dataset_id,
                "original_filename": record["original_filename"],
                "row_count": row_count,
                "column_count": col_count,
                "columns": columns,
                "descriptive_stats": stats,
                "sample": sample,
            },
        )
