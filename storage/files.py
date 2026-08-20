import shutil
from pathlib import Path

DATA_ROOT = Path("data")


def dataset_dir(session_id: str, dataset_id: str) -> Path:
    return DATA_ROOT / "sessions" / session_id / "datasets" / dataset_id


def save_dataset(session_id: str, dataset_id: str, source_path: str) -> str:
    dest_dir = dataset_dir(session_id, dataset_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "original.csv"
    shutil.copy2(source_path, dest)
    return str(dest)


def get_dataset_path(session_id: str, dataset_id: str) -> Path:
    return dataset_dir(session_id, dataset_id) / "original.csv"


def artifact_dir(session_id: str) -> Path:
    return DATA_ROOT / "sessions" / session_id / "artifacts"


def artifact_path(session_id: str, artifact_id: str, filename: str) -> Path:
    d = artifact_dir(session_id) / artifact_id
    d.mkdir(parents=True, exist_ok=True)
    return d / filename


def workspace_dir(session_id: str) -> Path:
    return DATA_ROOT / "sessions" / session_id / "workspace"
