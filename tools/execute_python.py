import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from executor.base import ExecutionService
from llm.base import ToolDefinition
from storage import database, files
from tools.base import Tool, ToolResult

# File extensions that should be registered as artifacts after execute_python runs
_ARTIFACT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".html", ".csv", ".json", ".parquet"}

# Map extension to artifact_type label
_ARTIFACT_TYPE = {
    ".png": "plot", ".jpg": "plot", ".jpeg": "plot", ".svg": "plot",
    ".html": "plot",
    ".csv": "table",
    ".json": "data",
    ".parquet": "data",
}

PREAMBLE = """\
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as _mpl
_mpl.rcParams['figure.figsize'] = (7, 4.5)
_mpl.rcParams['figure.dpi'] = 100
import seaborn as sns
import json, joblib, pathlib
import warnings
warnings.filterwarnings('ignore')
try:
    import shap
except ImportError:
    shap = None

def shap_explainer(model, X):
    # Avoids feature_names_in_ mismatches and categorical-split errors.
    # Always returns a 2D shap_values array (n_samples, n_features) ready for
    # summary_plot — handles both old-SHAP list format and new-SHAP 3D array.
    import shap as _shap
    X_arr = X.values if hasattr(X, 'values') else np.array(X)
    if hasattr(model, 'get_booster'):           # XGBoost sklearn API
        exp = _shap.TreeExplainer(
            model.get_booster(),
            feature_perturbation='tree_path_dependent',
        )
    else:
        exp = _shap.TreeExplainer(model)
    sv = exp.shap_values(X_arr)
    # Normalize to 2D: list format (older SHAP) or 3D ndarray (newer SHAP)
    if isinstance(sv, list):
        if len(sv) == 2:
            sv = sv[1]                              # binary: positive class
        else:
            sv = np.mean(np.abs(np.stack(sv, axis=-1)), axis=-1)  # multi-class: mean |SHAP|
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        if sv.shape[2] == 2:
            sv = sv[:, :, 1]                        # binary: positive class
        else:
            sv = np.abs(sv).mean(axis=-1)           # multi-class: mean |SHAP| across classes
    return exp, sv

def kernel_shap_explainer(model, X, background_size=100, explain_size=50):
    # Wraps predict_proba in a lambda so SHAP can't set read-only XGBoost properties.
    # Returns (explainer, shap_values_class1, X_subset) where shap_values_class1 is 2D.
    import shap as _shap
    X_arr = X.values if hasattr(X, 'values') else np.array(X)
    background = _shap.sample(X_arr, min(background_size, len(X_arr)))
    predict_fn = lambda x: model.predict_proba(x)
    kexp = _shap.KernelExplainer(predict_fn, background)
    n = min(explain_size, len(X_arr))
    ksv = kexp.shap_values(X_arr[:n])
    # KernelExplainer with predict_proba returns (n, features, n_classes) or list-per-class
    if isinstance(ksv, list):
        ksv_pos = ksv[1]
    else:
        ksv_pos = ksv[:, :, 1]
    return kexp, ksv_pos, X_arr[:n]

# ── Monkey-patch shap so direct calls always produce plot-ready 2D values ──────
# This fires even when the agent bypasses shap_explainer() and calls
# shap.TreeExplainer / shap.KernelExplainer directly.
if shap is not None:
    class _SV2D(np.ndarray):
        # 2D SHAP array whose integer subscript is a no-op.
        # sv[0] / sv[1] returns self — prevents class-index slicing into samples.
        def __new__(cls, arr):
            return np.asarray(arr).view(cls)
        def __getitem__(self, key):
            if isinstance(key, (int, np.integer)):
                return self   # sv[0] / sv[1] → return full 2D array unchanged
            return super().__getitem__(key)

    def _to2d(sv):
        # Collapse list / 3D shap values → 2D _SV2D (n_samples, n_features).
        if isinstance(sv, list):
            sv = sv[1] if len(sv) == 2 else np.mean(np.abs(np.stack(sv, axis=-1)), axis=-1)
        if isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = sv[:, :, 1] if sv.shape[2] == 2 else np.abs(sv).mean(axis=-1)
        return _SV2D(sv)

    _OrigTE = shap.TreeExplainer
    class _SafeTE:
        def __init__(self, model, *args, **kwargs):
            m = model.get_booster() if hasattr(model, 'get_booster') else model
            if 'model_output' not in kwargs:
                kwargs.setdefault('feature_perturbation', 'tree_path_dependent')
            self._e = _OrigTE(m, *args, **kwargs)
        def shap_values(self, X, *args, **kwargs):
            Xa = X.values if hasattr(X, 'values') else np.array(X)
            return _to2d(self._e.shap_values(Xa, *args, **kwargs))
        def __call__(self, X, *args, **kwargs):
            Xa = X.values if hasattr(X, 'values') else np.array(X)
            return self._e(Xa, *args, **kwargs)
        def __getattr__(self, name):
            return getattr(self._e, name)

    _OrigKE = shap.KernelExplainer
    class _SafeKE:
        def __init__(self, f, data, *args, **kwargs):
            _f = (lambda x: f(x)) if callable(f) else f
            Xbg = data.values if hasattr(data, 'values') else np.array(data)
            self._e = _OrigKE(_f, Xbg, *args, **kwargs)
        def shap_values(self, X, *args, **kwargs):
            Xa = X.values if hasattr(X, 'values') else np.array(X)
            return _to2d(self._e.shap_values(Xa, *args, **kwargs))
        def __getattr__(self, name):
            return getattr(self._e, name)

    _orig_splot = shap.summary_plot
    def _safe_splot(sv, features=None, feature_names=None, **kwargs):
        # np.asarray strips the _SV2D subclass so SHAP's internal indexing works correctly
        return _orig_splot(np.asarray(_to2d(sv)), features, feature_names=feature_names, **kwargs)

    shap.TreeExplainer   = _SafeTE
    shap.KernelExplainer = _SafeKE
    shap.summary_plot    = _safe_splot

"""

DESCRIPTION = """\
Execute Python code for data analysis. The following are pre-loaded and available:
- pd, np, plt, sns, json, joblib, pathlib: standard libraries
- df: pandas DataFrame for the given dataset_id (if provided)
- shap_explainer(model, X): returns (explainer, shap_values) via TreeExplainer — \
ALWAYS use this instead of calling shap.TreeExplainer directly. Avoids XGBoost \
feature_names_in_ / categorical-split errors. shap_values is always a 2D array \
(n_samples, n_features) ready for shap.summary_plot — multi-class models \
(e.g. Random Forest with 3+ classes) are automatically collapsed to mean |SHAP| \
across classes; binary models use the positive-class values.
- kernel_shap_explainer(model, X, background_size=100, explain_size=50): returns \
(explainer, shap_values_class1, X_subset) via KernelExplainer — ALWAYS use this \
instead of calling shap.KernelExplainer directly. Handles XGBoost incompatibilities \
and returns a ready-to-plot 2D array (n_samples, n_features) for the positive class. \
Never call shap.KernelExplainer(model.predict_proba, ...) yourself.

IMPORTANT: this runs as a subprocess, not a notebook — implicit expression evaluation \
does NOT print anything. You MUST call print() explicitly on every value you want to \
see. Wrong: `df.describe()`. Right: `print(df.describe())`. \
Save plots with plt.savefig('name.png') then plt.close() — do NOT use plt.show(). \
Generated files are captured as artifacts.

If you need a library that isn't imported by default (e.g. shap, scipy, plotly), \
pass it in the `packages` list and it will be installed before the code runs.\
"""

# Data-science packages that may be installed at runtime.
# Deliberately narrow — this is a whitelist, not a blacklist.
SAFE_PACKAGES: frozenset[str] = frozenset({
    "shap", "plotly", "plotly_express", "kaleido",
    "lightgbm", "catboost",
    "statsmodels", "scipy", "pingouin",
    "openpyxl", "xlrd", "pyarrow", "fastparquet",
    "missingno", "imbalanced-learn", "umap-learn",
    "yellowbrick", "eli5", "lime",
    "category-encoders", "feature-engine",
    "pyod", "tqdm", "tabulate",
})

# Canonical install name for packages whose import name differs from their pip name
_INSTALL_NAME: dict[str, str] = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "umap": "umap-learn",
    "imblearn": "imbalanced-learn",
}


def _install_packages(requested: list[str]) -> tuple[list[str], list[str]]:
    """Install whitelisted packages. Returns (installed, skipped_with_reason)."""
    installed: list[str] = []
    skipped: list[str] = []
    for pkg in requested:
        normalised = pkg.lower().replace("-", "_")
        canonical = _INSTALL_NAME.get(pkg, pkg)
        # Accept both the import name and the pip name in the whitelist
        if normalised in {p.replace("-", "_") for p in SAFE_PACKAGES}:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", canonical, "-q",
                 "--disable-pip-version-check"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                installed.append(pkg)
            else:
                skipped.append(f"{pkg} (install failed: {result.stderr.strip()[:120]})")
        else:
            skipped.append(f"{pkg} (not in the approved package list)")
    return installed, skipped


class ExecutePythonTool(Tool):
    name = "execute_python"
    description = DESCRIPTION

    def __init__(self, execution_service: ExecutionService, session_id: str):
        self.execution_service = execution_service
        self.session_id = session_id

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    },
                    "dataset_id": {
                        "type": "string",
                        "description": "Load this dataset as 'df' before running the code.",
                    },
                    "packages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Extra Python packages to install before running the code "
                            "(e.g. ['shap', 'scipy']). Only approved data-science packages "
                            "are allowed; others are silently skipped."
                        ),
                    },
                },
                "required": ["code"],
            },
        )

    def execute(self, **kwargs) -> ToolResult:
        code = kwargs.get("code", "").strip()
        dataset_id = kwargs.get("dataset_id")
        requested_packages: list[str] = kwargs.get("packages") or []

        if not code:
            return ToolResult(status="error", error_type="ValueError", message="code is required.")

        # Install requested packages before running
        installed, skipped = [], []
        if requested_packages:
            installed, skipped = _install_packages(requested_packages)

        preamble = PREAMBLE
        if dataset_id:
            record = database.get_dataset(dataset_id)
            if not record:
                return ToolResult(
                    status="error",
                    error_type="NotFound",
                    message=f"Dataset '{dataset_id}' not found.",
                )
            csv_path = Path(record["file_path"]).resolve()
            preamble += f'df = pd.read_csv(r"{csv_path}")\n\n'

        full_code = preamble + code
        working_dir = str(files.workspace_dir(self.session_id))
        result = self.execution_service.execute(full_code, working_dir)

        if result.status == "success":
            data: dict = {"stdout": result.stdout.strip(), "execution_time_ms": round(result.execution_time_ms)}

            # Move generated files into the artifacts directory and register in DB
            artifact_ids: list[str] = []
            for src_str in result.artifact_paths:
                src = Path(src_str)
                if src.suffix.lower() not in _ARTIFACT_EXTENSIONS:
                    continue
                art_id = f"art_{uuid.uuid4().hex[:8]}"
                dest = files.artifact_path(self.session_id, art_id, src.name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
                art_type = _ARTIFACT_TYPE.get(src.suffix.lower(), "file")
                database.register_artifact(
                    session_id=self.session_id,
                    artifact_type=art_type,
                    filename=src.name,
                    file_path=str(dest),
                    dataset_id=dataset_id,
                    description=f"Generated by execute_python: {src.name}",
                    artifact_id=art_id,
                )
                artifact_ids.append(art_id)

            if artifact_ids:
                data["artifact_ids"] = artifact_ids
            if installed:
                data["packages_installed"] = installed
            if skipped:
                data["packages_skipped"] = skipped
            return ToolResult(status="success", data=data)

        if result.status == "timeout":
            return ToolResult(
                status="error",
                error_type="TimeoutError",
                message=result.error_message,
            )

        return ToolResult(
            status="error",
            error_type=result.error_type or "RuntimeError",
            message=result.error_message,
            details=result.stderr[-1500:] if result.stderr else None,
        )
