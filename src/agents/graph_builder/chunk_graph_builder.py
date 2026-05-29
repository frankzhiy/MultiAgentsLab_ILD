"""C-3：异步逐 Chunk 建图（两阶段：先语义分析，再本体映射）。

流程：
  1. 读取配置（graph_builder.yaml）
  2. 加载两个 prompt 模板：分析 + 映射
  3. 对每个 AnnotatedChunk：
     a) LLM #1 → _ChunkAnalysis（自然语言列出 entities + relations）
     b) LLM #2 → _GraphBuilderOutput（把 analysis 映射到 NodeType/EdgeType）
  4. 组装 ChunkGraph

并发控制：所有 LLM 调用共享同一个 asyncio.Semaphore。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml
from src.agents.context_chunker.config_loader import resolve_project_path
from src.agents.graph_builder.clinical_context_preprocessor import (
    AnnotatedChunk,
    apply_context_graph_overrides,
    normalize_chunk_analysis,
)
from src.llm.async_instructor_client import call_with_instructor_async
from src.schemas.graph.clinical_graph import (
    ChunkGraph,
    _ChunkAnalysis,
    _GraphBuilderOutput,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/agents/graph_builder/graph_builder.yaml")
DEFAULT_ANALYSIS_PROMPT = "src/prompts/graph_builder/chunk_analysis_prompt.md"
DEFAULT_GRAPH_PROMPT = "src/prompts/graph_builder/chunk_graph_prompt.md"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = resolve_project_path(config_path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"graph_builder config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_prompt_template(prompt_path: str) -> str:
    p = resolve_project_path(prompt_path)
    if not p.exists():
        raise FileNotFoundError(f"prompt not found: {p}")
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _fill_template(template: str, values: dict[str, str]) -> str:
    """用 str.replace 填充 {key} 占位符（避免 .format 与 JSON 大括号冲突）。"""
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", value)
    return out


def _build_analysis_prompt(template: str, ac: AnnotatedChunk) -> str:
    chunk = ac.chunk
    return _fill_template(template, {
        "chunk_id": chunk.chunk_id,
        "section_name": chunk.section_name,
        "section_index": str(chunk.section_index),
        "chunk_type": chunk.chunk_type,
        "annotated_text": ac.format_for_prompt(),
    })


def _build_mapping_prompt(
    template: str,
    ac: AnnotatedChunk,
    analysis: _ChunkAnalysis,
) -> str:
    chunk = ac.chunk
    analysis_json = json.dumps(
        analysis.model_dump(), ensure_ascii=False, indent=2,
    )
    return _fill_template(template, {
        "chunk_id": chunk.chunk_id,
        "section_name": chunk.section_name,
        "section_index": str(chunk.section_index),
        "chunk_type": chunk.chunk_type,
        "annotated_text": ac.format_for_prompt(),
        "analysis_json": analysis_json,
    })


# ---------------------------------------------------------------------------
# Single chunk builder（两阶段）
# ---------------------------------------------------------------------------

def _safe_artifact_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value)


def _write_json_artifact(
    output_dir: Path | None,
    chunk_id: str,
    stage: str,
    payload: dict[str, Any],
) -> None:
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_safe_artifact_name(chunk_id)}_{stage}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _empty_graph(ac: AnnotatedChunk, warning: str) -> ChunkGraph:
    chunk = ac.chunk
    return ChunkGraph(
        chunk_id=chunk.chunk_id,
        case_id=chunk.case_id,
        section_id=chunk.section_id,
        section_index=chunk.section_index,
        chunk_index=chunk.chunk_index,
        chunk_type=chunk.chunk_type,
        chunk_text=chunk.text,
        validation_warnings=[warning],
    )


async def _run_analysis(
    ac: AnnotatedChunk,
    analysis_template: str,
    builder_cfg: dict[str, Any],
    semaphore: asyncio.Semaphore,
    debug_output_dir: Path | None = None,
) -> _ChunkAnalysis | None:
    """C-3a：纯语义分析，输出 entities + relations 的自然语言描述。"""
    prompt = _build_analysis_prompt(analysis_template, ac)
    async with semaphore:
        try:
            result = await call_with_instructor_async(
                response_model=_ChunkAnalysis,
                messages=[{"role": "user", "content": prompt}],
                model=builder_cfg["model"],
                temperature=builder_cfg.get("temperature", 0.0),
                max_tokens=builder_cfg.get("analysis_max_tokens", 2048),
                top_p=builder_cfg.get("top_p", 1.0),
                timeout=builder_cfg.get("timeout", 120),
                max_retries=builder_cfg.get("max_retries", 3),
            )
            result = normalize_chunk_analysis(ac, result)
            _write_json_artifact(
                debug_output_dir,
                ac.chunk.chunk_id,
                "c3a_analysis",
                {
                    "stage": "C-3a",
                    "chunk_id": ac.chunk.chunk_id,
                    "chunk_text": ac.chunk.text,
                    "output": result.model_dump(),
                },
            )
            return result
        except Exception as exc:
            logger.warning(
                "chunk %s analysis (C-3a) failed: %s",
                ac.chunk.chunk_id, exc,
            )
            _write_json_artifact(
                debug_output_dir,
                ac.chunk.chunk_id,
                "c3a_analysis_error",
                {
                    "stage": "C-3a",
                    "chunk_id": ac.chunk.chunk_id,
                    "chunk_text": ac.chunk.text,
                    "error": str(exc),
                },
            )
            return None


async def _run_mapping(
    ac: AnnotatedChunk,
    analysis: _ChunkAnalysis,
    mapping_template: str,
    builder_cfg: dict[str, Any],
    semaphore: asyncio.Semaphore,
    debug_output_dir: Path | None = None,
) -> _GraphBuilderOutput | None:
    """C-3b：根据 analysis 把 entities/relations 映射到 NodeType/EdgeType。"""
    prompt = _build_mapping_prompt(mapping_template, ac, analysis)
    async with semaphore:
        try:
            result = await call_with_instructor_async(
                response_model=_GraphBuilderOutput,
                messages=[{"role": "user", "content": prompt}],
                model=builder_cfg["model"],
                temperature=builder_cfg.get("temperature", 0.0),
                max_tokens=builder_cfg.get("max_tokens", 2048),
                top_p=builder_cfg.get("top_p", 1.0),
                timeout=builder_cfg.get("timeout", 120),
                max_retries=builder_cfg.get("max_retries", 3),
            )
            _write_json_artifact(
                debug_output_dir,
                ac.chunk.chunk_id,
                "c3b_mapping",
                {
                    "stage": "C-3b",
                    "chunk_id": ac.chunk.chunk_id,
                    "chunk_text": ac.chunk.text,
                    "analysis": analysis.model_dump(),
                    "output": result.model_dump(),
                },
            )
            return result
        except Exception as exc:
            logger.warning(
                "chunk %s mapping (C-3b) failed: %s",
                ac.chunk.chunk_id, exc,
            )
            _write_json_artifact(
                debug_output_dir,
                ac.chunk.chunk_id,
                "c3b_mapping_error",
                {
                    "stage": "C-3b",
                    "chunk_id": ac.chunk.chunk_id,
                    "chunk_text": ac.chunk.text,
                    "analysis": analysis.model_dump(),
                    "error": str(exc),
                },
            )
            return None


async def _build_chunk_graph(
    ac: AnnotatedChunk,
    analysis_template: str,
    mapping_template: str,
    builder_cfg: dict[str, Any],
    semaphore: asyncio.Semaphore,
    debug_output_dir: Path | None = None,
) -> ChunkGraph:
    chunk = ac.chunk

    # 阶段 a：语义分析
    analysis = await _run_analysis(
        ac,
        analysis_template,
        builder_cfg,
        semaphore,
        debug_output_dir=debug_output_dir,
    )
    if analysis is None:
        return _empty_graph(ac, "C-3a 语义分析 LLM 调用失败")

    # 阶段 b：本体映射
    result = await _run_mapping(
        ac,
        analysis,
        mapping_template,
        builder_cfg,
        semaphore,
        debug_output_dir=debug_output_dir,
    )
    if result is None:
        return _empty_graph(ac, "C-3b 本体映射 LLM 调用失败")

    graph = ChunkGraph(
        chunk_id=chunk.chunk_id,
        case_id=chunk.case_id,
        section_id=chunk.section_id,
        section_index=chunk.section_index,
        chunk_index=chunk.chunk_index,
        chunk_type=chunk.chunk_type,
        chunk_text=chunk.text,
        nodes=result.nodes,
        edges=result.edges,
    )
    apply_context_graph_overrides(ac, graph)
    return graph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def build_chunk_graphs_async(
    annotated_chunks: list[AnnotatedChunk],
    config_path: Path | None = None,
    semaphore: asyncio.Semaphore | None = None,
    debug_output_dir: Path | None = None,
) -> list[ChunkGraph]:
    """对一批 AnnotatedChunk 并发建图（两阶段：分析 → 映射）。

    Args:
        semaphore: 外部传入时复用（跨 section 全局限速），为 None 时自行创建。
        debug_output_dir: 若提供，则保存每个 chunk 的 C-3a/C-3b 原始结构化输出。
    """
    cfg = _load_config(config_path)
    builder_cfg: dict[str, Any] = cfg.get("chunk_graph_builder", {})
    analysis_template = _load_prompt_template(
        builder_cfg.get("analysis_prompt_path", DEFAULT_ANALYSIS_PROMPT)
    )
    mapping_template = _load_prompt_template(
        builder_cfg.get("prompt_path", DEFAULT_GRAPH_PROMPT)
    )
    if semaphore is None:
        semaphore = asyncio.Semaphore(builder_cfg.get("max_concurrency", 4))

    tasks = [
        _build_chunk_graph(
            ac,
            analysis_template,
            mapping_template,
            builder_cfg,
            semaphore,
            debug_output_dir=debug_output_dir,
        )
        for ac in annotated_chunks
    ]
    return await asyncio.gather(*tasks)
