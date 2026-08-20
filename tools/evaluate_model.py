import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from llm.base import ToolDefinition
from storage import database
from tools.base import Tool, ToolResult
from tools.run_model import SUPPORTED_MODELS, MODEL_KWARGS, _prepare_features

REGRESSION_SCORING = {
    "r2":   "r2",
    "mae":  "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
}
CLASSIFICATION_SCORING = {
    "accuracy":  "accuracy",
    "f1":        "f1_weighted",
    "precision": "precision_weighted",
    "recall":    "recall_weighted",
}


class EvaluateModelTool(Tool):
    name = "evaluate_model"
    description = (
        "Evaluate a model with k-fold cross-validation for robust performance estimates. "
        "Returns mean ± std for each metric across folds. "
        "Use when comparing models or when a reliable performance estimate matters more than "
        "saving a trained model. Does not save a model artifact."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "dataset_id":      {"type": "string"},
                    "target_column":   {"type": "string"},
                    "model_type":      {"type": "string", "enum": list(SUPPORTED_MODELS)},
                    "feature_columns": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Columns to use as features. Defaults to all except target.",
                    },
                    "cv_folds": {
                        "type": "integer",
                        "description": "Number of folds (default 5).",
                    },
                },
                "required": ["dataset_id", "target_column", "model_type"],
            },
        )

    def execute(self, **kwargs) -> ToolResult:
        dataset_id    = kwargs.get("dataset_id")
        target_column = kwargs.get("target_column")
        model_type    = kwargs.get("model_type", "").lower()
        feature_cols  = kwargs.get("feature_columns")
        cv_folds      = int(kwargs.get("cv_folds", 5))

        if model_type not in SUPPORTED_MODELS:
            return ToolResult(status="error", error_type="ValueError",
                              message=f"model_type must be one of: {list(SUPPORTED_MODELS)}")

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

        try:
            X, y, feature_names = _prepare_features(df, feature_cols, target_column)
        except Exception as e:
            return ToolResult(status="error", error_type=type(e).__name__, message=str(e))

        ModelClass, task = SUPPORTED_MODELS[model_type]
        model = ModelClass(**MODEL_KWARGS[model_type])

        scoring = REGRESSION_SCORING if task == "regression" else CLASSIFICATION_SCORING
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42) if task == "regression" \
             else StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        try:
            cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        except Exception as e:
            return ToolResult(status="error", error_type=type(e).__name__, message=str(e))

        metrics: dict = {}
        for metric_name, score_key in scoring.items():
            raw = cv_results[f"test_{score_key}"]
            # neg_ scores need sign flip
            sign = -1 if score_key.startswith("neg_") else 1
            vals = raw * sign
            metrics[metric_name] = {
                "mean": round(float(vals.mean()), 4),
                "std":  round(float(vals.std()), 4),
                "per_fold": [round(float(v), 4) for v in vals],
            }

        return ToolResult(
            status="success",
            data={
                "model_type":      model_type,
                "task":            task,
                "target_column":   target_column,
                "cv_folds":        cv_folds,
                "n_samples":       len(X),
                "n_features":      len(feature_names),
                "cv_metrics":      metrics,
                "interpretation": (
                    f"{model_type} cross-validated over {cv_folds} folds on {len(X)} samples. "
                    + (f"Mean R²={metrics['r2']['mean']} ± {metrics['r2']['std']}."
                       if task == "regression"
                       else f"Mean accuracy={metrics['accuracy']['mean']} ± {metrics['accuracy']['std']}, "
                            f"F1={metrics['f1']['mean']} ± {metrics['f1']['std']}.")
                ),
            },
        )
