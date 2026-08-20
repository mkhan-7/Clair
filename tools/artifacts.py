from pathlib import Path

from llm.base import ToolDefinition
from storage import database
from tools.base import Tool, ToolResult


class ListArtifactsTool(Tool):
    name = "list_artifacts"
    description = "List all artifacts (plots, tables, models) generated in the current session."

    def __init__(self, session_id: str):
        self.session_id = session_id

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def execute(self, **kwargs) -> ToolResult:
        artifacts = database.get_session_artifacts(self.session_id)
        slim = [
            {
                "artifact_id": a["id"],
                "type": a["artifact_type"],
                "filename": a["filename"],
                "file_path": str(Path(a["file_path"]).resolve()),
                "description": a["description"],
                "analysis_id": a["analysis_id"],
                "dataset_id": a["dataset_id"],
                "created_at": a["created_at"],
            }
            for a in artifacts
        ]
        return ToolResult(status="success", data={"artifacts": slim, "count": len(slim)})


class GetArtifactTool(Tool):
    name = "get_artifact"
    description = "Retrieve metadata for a specific artifact by its ID."

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "artifact_id": {
                        "type": "string",
                        "description": "The artifact ID to retrieve.",
                    }
                },
                "required": ["artifact_id"],
            },
        )

    def execute(self, **kwargs) -> ToolResult:
        artifact_id = kwargs.get("artifact_id")
        record = database.get_artifact(artifact_id)
        if not record:
            return ToolResult(status="error", error_type="NotFound",
                              message=f"Artifact '{artifact_id}' not found.")
        return ToolResult(status="success", data=dict(record))
