import pandas as pd
import numpy as np

from llm.base import ToolDefinition
from storage import database
from tools.base import Tool, ToolResult
from tools.run_model import SUPPORTED_MODELS


def _infer_task(col: pd.Series) -> tuple[str, dict]:
    """Return ('regression'|'classification', target_analysis_dict)."""
    n = len(col)
    n_unique = int(col.nunique())
    dtype = str(col.dtype)

    analysis: dict = {
        "dtype": dtype,
        "unique_values": n_unique,
    }

    # Clearly categorical
    if col.dtype == object or n_unique <= 20:
        vc = col.value_counts(normalize=True).round(4)
        analysis["class_distribution"] = {str(k): float(v) for k, v in vc.items()}
        majority = float(vc.iloc[0])
        analysis["is_balanced"] = majority < 0.75
        analysis["majority_class_pct"] = round(majority * 100, 1)
        return "classification", analysis

    # Continuous numeric
    analysis["mean"]   = round(float(col.mean()), 4)
    analysis["std"]    = round(float(col.std()), 4)
    analysis["min"]    = round(float(col.min()), 4)
    analysis["max"]    = round(float(col.max()), 4)
    return "regression", analysis


def _rank_models(task: str, n_rows: int, n_features: int, is_balanced: bool) -> list[dict]:
    """Return ranked model recommendations based on dataset characteristics."""

    if task == "regression":
        recs = [
            {
                "model_type": "xgboost_regressor",
                "rank": 1,
                "recommended": True,
                "best_for": "predictive accuracy",
                "reasoning": (
                    f"Gradient boosting typically achieves best out-of-box performance "
                    f"on tabular data. With {n_rows} rows and hyperparameter tuning it "
                    f"should generalise well."
                ),
                "trade_offs": "Less interpretable than linear regression.",
            },
            {
                "model_type": "random_forest_regressor",
                "rank": 2,
                "recommended": False,
                "best_for": "robust baseline with feature importance",
                "reasoning": "Strong ensemble baseline, robust to outliers and noisy features.",
                "trade_offs": "Slower than XGBoost; can overfit on small datasets.",
            },
            {
                "model_type": "linear_regression",
                "rank": 3,
                "recommended": False,
                "best_for": "interpretability — when coefficient magnitudes matter",
                "reasoning": "Fast, fully interpretable, ideal when explaining relationships matters more than accuracy.",
                "trade_offs": "Assumes linear relationships; underperforms when interactions are non-linear.",
            },
        ]
        # Prefer linear model on small datasets
        if n_rows < 300:
            recs[0]["recommended"] = False
            recs[2]["recommended"] = True
            recs[2]["rank"] = 1
            recs[0]["rank"] = 3
            recs.sort(key=lambda x: x["rank"])

    else:  # classification
        recs = [
            {
                "model_type": "xgboost_classifier",
                "rank": 1,
                "recommended": True,
                "best_for": "predictive accuracy",
                "reasoning": (
                    f"Best default for tabular classification with {n_rows} rows. "
                    f"Handles mixed feature types well and benefits from tuning."
                ),
                "trade_offs": "Less interpretable; predictions harder to explain to stakeholders.",
            },
            {
                "model_type": "random_forest_classifier",
                "rank": 2,
                "recommended": False,
                "best_for": "robust baseline with feature importance",
                "reasoning": "Strong ensemble, good feature importance, less sensitive to hyperparameters than XGBoost.",
                "trade_offs": "Slightly lower accuracy ceiling than gradient boosting.",
            },
            {
                "model_type": "logistic_regression",
                "rank": 3,
                "recommended": False,
                "best_for": "interpretability — when odds ratios or coefficients matter",
                "reasoning": "Fully interpretable coefficients, fast, well-calibrated probabilities.",
                "trade_offs": "Assumes linear decision boundary; weaker on complex non-linear patterns.",
            },
        ]
        # Prefer logistic regression on small datasets or when likely linear
        if n_rows < 300:
            recs[0]["recommended"] = False
            recs[2]["recommended"] = True
            recs[2]["rank"] = 1
            recs[0]["rank"] = 3
            recs.sort(key=lambda x: x["rank"])

    # Filter out models not available (e.g. XGBoost missing libomp)
    available = set(SUPPORTED_MODELS.keys())
    recs = [r for r in recs if r["model_type"] in available]
    return recs


class RecommendModelTool(Tool):
    name = "recommend_model"
    description = (
        "Analyse a dataset and target column to recommend the most appropriate model type "
        "before training. Call this when the user asks to 'predict X' or 'build a model' "
        "without specifying which algorithm to use. Returns ranked recommendations with reasoning."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "dataset_id":    {"type": "string", "description": "Dataset to analyse."},
                    "target_column": {"type": "string", "description": "Column to predict."},
                    "feature_columns": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Planned feature columns (optional — used to count features).",
                    },
                },
                "required": ["dataset_id", "target_column"],
            },
        )

    def execute(self, **kwargs) -> ToolResult:
        dataset_id     = kwargs.get("dataset_id")
        target_column  = kwargs.get("target_column")
        feature_cols   = kwargs.get("feature_columns")

        record = database.get_dataset(dataset_id)
        if not record:
            return ToolResult(status="error", error_type="NotFound",
                              message=f"Dataset '{dataset_id}' not found.")

        try:
            df = pd.read_csv(record["file_path"])
        except Exception as e:
            return ToolResult(status="error", error_type=type(e).__name__, message=str(e))

        if target_column not in df.columns:
            return ToolResult(status="error", error_type="ColumnNotFound",
                              message=f"Column '{target_column}' not found.",
                              details=f"Available: {', '.join(df.columns)}")

        col = df[target_column].dropna()
        task, target_analysis = _infer_task(col)

        n_rows = len(df)
        candidate_features = feature_cols or [c for c in df.columns if c != target_column]
        n_features = len(candidate_features)

        is_balanced = target_analysis.get("is_balanced", True)
        recs = _rank_models(task, n_rows, n_features, is_balanced)

        notes: list[str] = []
        if task == "classification":
            majority_pct = target_analysis.get("majority_class_pct", 50)
            if majority_pct > 75:
                notes.append(
                    f"Class imbalance detected ({majority_pct}% majority class). "
                    "Consider class_weight='balanced' or resampling before training."
                )
            else:
                notes.append(
                    f"Classes are reasonably balanced ({majority_pct}% majority). "
                    "No resampling needed."
                )
        if n_rows < 500:
            notes.append(
                f"Small dataset ({n_rows} rows). Simpler models may generalise better. "
                "Cross-validation (evaluate_model) is more reliable than a single train/test split."
            )

        return ToolResult(
            status="success",
            data={
                "task":             task,
                "target_column":    target_column,
                "target_analysis":  target_analysis,
                "dataset_rows":     n_rows,
                "planned_features": n_features,
                "recommendations":  recs,
                "notes":            notes,
                "suggested_next_step": (
                    f"Call run_model with model_type='{recs[0]['model_type']}', "
                    f"target_column='{target_column}', tune_hyperparameters=true."
                ),
            },
        )
