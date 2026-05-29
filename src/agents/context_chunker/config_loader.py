from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "agents" / "context_chunker.yaml"


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_context_chunker_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"context_chunker config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    required_top = {"agent_name", "sentence_classifier", "reviewer", "chunk_splitter"}
    missing = sorted(required_top - set(config))
    if missing:
        raise ValueError(
            f"context_chunker config missing required fields: {', '.join(missing)}"
        )

    required_phase_fields = {"model", "temperature", "max_tokens", "top_p", "prompt_path"}
    for phase_key in ("sentence_classifier", "reviewer", "chunk_splitter"):
        phase_cfg = config.get(phase_key) or {}
        miss = sorted(required_phase_fields - set(phase_cfg))
        if miss:
            raise ValueError(
                f"context_chunker config '{phase_key}' missing fields: {', '.join(miss)}"
            )

    return config
