"""Artifact storage with standardized layout."""

from cbe.artifacts.local_store import LocalArtifactStore


def create_artifact_store(config) -> LocalArtifactStore:
    """Create an artifact store from config."""
    return LocalArtifactStore(
        output_dir=config.output_dir,
        max_checkpoints=config.training.max_checkpoints,
    )
