import json
import uuid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from joblib import dump as joblib_dump
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)

from llm.base import ToolDefinition
from storage import database, files
from tools.base import Tool, ToolResult

try:
    from xgboost import XGBRegressor, XGBClassifier
    _XGB_AVAILABLE = True
except Exception:
    _XGB_AVAILABLE = False

SUPPORTED_MODELS: dict = {
    "linear_regression":          (LinearRegression,        "regression"),
    "random_forest_regressor":    (RandomForestRegressor,   "regression"),
    "logistic_regression":        (LogisticRegression,      "classification"),
    "random_forest_classifier":   (RandomForestClassifier,  "classification"),
}
if _XGB_AVAILABLE:
    SUPPORTED_MODELS["xgboost_regressor"]   = (XGBRegressor,   "regression")
    SUPPORTED_MODELS["xgboost_classifier"]  = (XGBClassifier,  "classification")

MODEL_KWARGS: dict = {
    "linear_regression":         {},
    "random_forest_regressor":   {"random_state": 42, "n_estimators": 100},
    "logistic_regression":       {"random_state": 42, "max_iter": 1000},
    "random_forest_classifier":  {"random_state": 42, "n_estimators": 100},
    "xgboost_regressor":         {"random_state": 42, "n_estimators": 100, "verbosity": 0},
    "xgboost_classifier":        {"random_state": 42, "n_estimators": 100, "verbosity": 0},
}

PARAM_GRIDS: dict = {
    "random_forest_regressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "random_forest_classifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "logistic_regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
    },
    "xgboost_regressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.3],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
    "xgboost_classifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.3],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
    "linear_regression": {},  # nothing to tune
}


def _prepare_features(
    df: pd.DataFrame, feature_columns: list[str] | None, target_column: str
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    cols = feature_columns if feature_columns else [c for c in df.columns if c != target_column]
    cols = [c for c in cols if c in df.columns and c != target_column]

    data = df[cols + [target_column]].dropna(subset=[target_column])
    X = pd.get_dummies(data[cols], drop_first=False)
    # pandas 2.0+ returns bool dtype for indicator columns; cast everything to float
    X = X.astype(float)
    X = X.fillna(X.median())
    y = data[target_column]
    return X, y, list(X.columns)


def _regression_metrics(y_true, y_pred) -> dict:
    return {
        "r2":   round(float(r2_score(y_true, y_pred)), 4),
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(mean_squared_error(y_true, y_pred) ** 0.5), 4),
    }


def _classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    is_binary = len(np.unique(y_true)) == 2
    avg = "binary" if is_binary else "weighted"
    m = {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4),
    }
    if y_prob is not None:
        try:
            if is_binary:
                m["roc_auc"] = round(float(roc_auc_score(y_true, y_prob[:, 1])), 4)
            else:
                m["roc_auc"] = round(float(roc_auc_score(y_true, y_prob, multi_class="ovr")), 4)
        except Exception:
            pass
    return m


class RunModelTool(Tool):
    name = "run_model"
    description = (
        "Train a predictive model on a dataset. "
        f"Supported model types: {', '.join(SUPPORTED_MODELS)}. "
        "Performs an 80/20 train/test split, returns evaluation metrics, and saves "
        "the trained model and visualizations as artifacts."
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
                    "dataset_id":      {"type": "string", "description": "Dataset to model."},
                    "target_column":   {"type": "string", "description": "Column to predict."},
                    "model_type":      {"type": "string", "enum": list(SUPPORTED_MODELS)},
                    "feature_columns": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Columns to use as features. Defaults to all columns except target.",
                    },
                    "test_size": {
                        "type": "number",
                        "description": "Fraction of data for testing (default 0.2).",
                    },
                    "tune_hyperparameters": {
                        "type": "boolean",
                        "description": "Run RandomizedSearchCV to find better hyperparameters (default true). Set false for a quick baseline.",
                    },
                    "n_iter": {
                        "type": "integer",
                        "description": "Number of random parameter combinations to try when tuning (default 10).",
                    },
                    "cv_folds": {
                        "type": "integer",
                        "description": "Number of cross-validation folds when tuning (default 5).",
                    },
                },
                "required": ["dataset_id", "target_column", "model_type"],
            },
        )

    def execute(self, **kwargs) -> ToolResult:
        dataset_id    = kwargs.get("dataset_id")
        target_column = kwargs.get("target_column")
        model_type    = kwargs.get("model_type", "").lower()
        feature_cols        = kwargs.get("feature_columns")
        test_size           = float(kwargs.get("test_size", 0.2))
        tune_hyperparameters = kwargs.get("tune_hyperparameters", True)
        n_iter              = int(kwargs.get("n_iter", 10))
        cv_folds            = int(kwargs.get("cv_folds", 5))

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
                              message=f"Target column '{target_column}' not found.",
                              details=f"Available: {', '.join(df.columns)}")

        try:
            X, y, final_feature_names = _prepare_features(df, feature_cols, target_column)
        except Exception as e:
            return ToolResult(status="error", error_type=type(e).__name__, message=str(e))

        if len(X) < 10:
            return ToolResult(status="error", error_type="InsufficientData",
                              message="Not enough rows to train a model.")

        ModelClass, task = SUPPORTED_MODELS[model_type]
        model_kwargs = MODEL_KWARGS[model_type]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        scoring = "r2" if task == "regression" else (
            "roc_auc" if len(np.unique(y)) == 2 else "f1_weighted"
        )
        best_params: dict | None = None

        try:
            param_grid = PARAM_GRIDS.get(model_type, {})
            if tune_hyperparameters and param_grid:
                search = RandomizedSearchCV(
                    ModelClass(**model_kwargs),
                    param_grid,
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring=scoring,
                    random_state=42,
                    n_jobs=-1,
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
                best_params = search.best_params_
            else:
                model = ModelClass(**model_kwargs)
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        except Exception as e:
            return ToolResult(status="error", error_type=type(e).__name__, message=str(e))

        # Metrics
        if task == "regression":
            metrics = _regression_metrics(y_test, y_pred)
        else:
            metrics = _classification_metrics(y_test, y_pred, y_prob)

        # Feature importances
        feature_importance: dict = {}
        if hasattr(model, "feature_importances_"):
            fi = dict(zip(final_feature_names, model.feature_importances_.tolist()))
            feature_importance = dict(sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[:20])
        elif hasattr(model, "coef_"):
            coef = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
            fi = dict(zip(final_feature_names, coef.tolist()))
            feature_importance = dict(sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[:20])

        # Create analysis record (summary filled after saving artifacts)
        analysis_id = database.create_analysis(
            session_id=self.session_id,
            dataset_id=dataset_id,
            analysis_type=model_type,
            summary="",
        )

        artifact_ids: list[str] = []

        # Save model pickle
        model_filename = f"{model_type}_{analysis_id}.pkl"
        model_path = files.artifact_path(self.session_id, f"art_{uuid.uuid4().hex[:8]}", model_filename)
        joblib_dump(model, model_path)
        art_id = database.register_artifact(
            session_id=self.session_id, dataset_id=dataset_id, analysis_id=analysis_id,
            artifact_type="model", filename=model_filename, file_path=str(model_path),
            description=f"Trained {model_type} model",
        )
        artifact_ids.append(art_id)

        # Save companion model_info.json so the agent can reproduce preprocessing exactly
        # (feature names are needed for SHAP and other post-hoc analyses)
        info = {
            "model_type": model_type,
            "task": task,
            "target_column": target_column,
            "feature_columns": final_feature_names,
            "dataset_id": dataset_id,
            "analysis_id": analysis_id,
            "model_pkl": str(model_path),
        }
        info_filename = model_filename.replace(".pkl", "_info.json")
        info_path = model_path.parent / info_filename
        info_path.write_text(json.dumps(info, indent=2))
        info_art_id = database.register_artifact(
            session_id=self.session_id, dataset_id=dataset_id, analysis_id=analysis_id,
            artifact_type="model_info", filename=info_filename, file_path=str(info_path),
            description=f"Feature list and metadata for {model_type} ({analysis_id})",
        )
        artifact_ids.append(info_art_id)

        # Visualizations
        if feature_importance:
            art_id = self._save_feature_importance(feature_importance, analysis_id, dataset_id, model_type)
            artifact_ids.append(art_id)

        if task == "regression":
            art_id = self._save_predicted_vs_actual(y_test, y_pred, target_column, analysis_id, dataset_id)
            artifact_ids.append(art_id)
            art_id = self._save_residual_plot(y_test, y_pred, target_column, analysis_id, dataset_id)
            artifact_ids.append(art_id)
        else:
            labels = sorted(y.unique())
            art_id = self._save_confusion_matrix(y_test, y_pred, labels, analysis_id, dataset_id)
            artifact_ids.append(art_id)
            if y_prob is not None and len(labels) == 2:
                art_id = self._save_roc_curve(y_test, y_prob[:, 1], analysis_id, dataset_id)
                artifact_ids.append(art_id)

        # Update analysis summary
        feat_str = ", ".join(list(feature_importance.keys())[:5])
        if task == "regression":
            summary = (f"{model_type} on {dataset_id} (target: {target_column}): "
                       f"R²={metrics['r2']}, MAE={metrics['mae']}, RMSE={metrics['rmse']}. "
                       f"Top features: {feat_str}.")
        else:
            summary = (f"{model_type} on {dataset_id} (target: {target_column}): "
                       f"accuracy={metrics['accuracy']}, F1={metrics['f1']}, "
                       f"ROC-AUC={metrics.get('roc_auc', 'n/a')}. "
                       f"Top features: {feat_str}.")

        with database._connect() as conn:
            conn.execute("UPDATE analyses SET summary = ? WHERE id = ?", (summary, analysis_id))

        return ToolResult(
            status="success",
            data={
                "analysis_id": analysis_id,
                "model_type": model_type,
                "task": task,
                "target_column": target_column,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "features_used": final_feature_names,
                "metrics": metrics,
                "best_params": best_params,
                "tuning_used": tune_hyperparameters and bool(PARAM_GRIDS.get(model_type)),
                "top_feature_importance": feature_importance,
                "artifact_ids": artifact_ids,
                "summary": summary,
            },
        )

    # ── Visualization helpers ──────────────────────────────────────────────────

    def _register_plot(self, fig, filename, description, analysis_id, dataset_id) -> str:
        art_id = f"art_{uuid.uuid4().hex[:8]}"
        path = files.artifact_path(self.session_id, art_id, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        database.register_artifact(
            session_id=self.session_id, dataset_id=dataset_id, analysis_id=analysis_id,
            artifact_type="plot", filename=filename, file_path=str(path),
            description=description, artifact_id=art_id,
        )
        return art_id

    def _save_feature_importance(self, fi: dict, analysis_id, dataset_id, model_type) -> str:
        names = list(fi.keys())[:15]
        values = [fi[n] for n in names]
        colors = ["steelblue" if v >= 0 else "tomato" for v in values]

        fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.45)))
        ax.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor="white")
        ax.set_xlabel("Importance / Coefficient")
        ax.set_title(f"Feature Importance — {model_type}")
        plt.tight_layout()
        return self._register_plot(fig, f"feature_importance_{analysis_id}.png",
                                   "Feature importance plot", analysis_id, dataset_id)

    def _save_predicted_vs_actual(self, y_true, y_pred, target, analysis_id, dataset_id) -> str:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_true, y_pred, alpha=0.4, s=20, color="steelblue")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel(f"Actual {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title("Predicted vs Actual")
        ax.legend()
        plt.tight_layout()
        return self._register_plot(fig, f"predicted_vs_actual_{analysis_id}.png",
                                   "Predicted vs actual plot", analysis_id, dataset_id)

    def _save_residual_plot(self, y_true, y_pred, target, analysis_id, dataset_id) -> str:
        residuals = np.array(y_true) - np.array(y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_pred, residuals, alpha=0.4, s=20, color="steelblue")
        ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
        ax.set_xlabel(f"Predicted {target}")
        ax.set_ylabel("Residual")
        ax.set_title("Residual Plot")
        plt.tight_layout()
        return self._register_plot(fig, f"residuals_{analysis_id}.png",
                                   "Residual plot", analysis_id, dataset_id)

    def _save_confusion_matrix(self, y_true, y_pred, labels, analysis_id, dataset_id) -> str:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels) - 1)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        return self._register_plot(fig, f"confusion_matrix_{analysis_id}.png",
                                   "Confusion matrix", analysis_id, dataset_id)

    def _save_roc_curve(self, y_true, y_score, analysis_id, dataset_id) -> str:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"ROC (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Random classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        plt.tight_layout()
        return self._register_plot(fig, f"roc_curve_{analysis_id}.png",
                                   "ROC curve", analysis_id, dataset_id)
