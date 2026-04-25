from __future__ import annotations

from .data_collection.config import requires_masking as _requires_data_collection_masking
from .data_collection.data_collection import DataCollectionEnvironmentConfig


def requires_masking(config: dict) -> bool:
    """Return whether the active environment configuration requires agent masking."""
    env_config = config["environment"].copy()
    env_config.pop("sequential_obs", None)
    return _requires_data_collection_masking(DataCollectionEnvironmentConfig(**env_config))
