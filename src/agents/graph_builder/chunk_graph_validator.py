"""C-4：ChunkGraph 验证器。

验证分三层：

  HARD-节点：GraphNode.source_text 必须是 chunk_text 的精确子串
    → 不满足：删除节点及其关联边，写入 validation_warnings

  HARD-边：边引用的 source_node_id / target_node_id 必须存在
    → 悬空边直接删除

  SOFT-边（LLM 二次审查）：满足任意条件时触发
    a. edge.confidence < confidence_threshold
    b. edge.source_text 在 chunk_text 中完全找不到
    LLM 输入：边的 reasoning，输出 EdgeValidationResult
    verdict=invalid → 删除边
    verdict=low_confidence → 保留但 low_confidence_edge_ids 标记
    verdict=valid → 不变
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import yaml
from src.agents.context_chunker.config_loader import resolve_project_path
from src.agents.graph_builder.patient_anchor import (
    ensure_patient_anchor,
    is_implicit_patient_node,
)
from src.llm.async_instructor_client import call_with_instructor_async
from src.schemas.graph.clinical_graph import (
    EDGE_ENDPOINT_CONSTRAINTS,
    ChunkGraph,
    EdgeValidationResult,
    GraphEdge,
    GraphNode,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/agents/graph_builder/graph_builder.yaml")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _get_validator_cfg(config_path: Path | None = None) -> dict[str, Any]:
    path = resolve_project_path(config_path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"graph_builder config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    builder_cfg = raw.get("chunk_graph_builder", {})
    validator_cfg = raw.get("edge_validator", {})
    # confidence_threshold 可在 builder 或 validator 配置块中设置
    threshold = validator_cfg.get(
        "confidence_threshold",
        builder_cfg.get("confidence_threshold", 0.5),
    )
    validator_cfg["confidence_threshold"] = threshold
    return validator_cfg


# ---------------------------------------------------------------------------
# HARD rule: node source_text must be exact substring of chunk_text
# ---------------------------------------------------------------------------

def _validate_nodes_hard(graph: ChunkGraph) -> tuple[list[GraphNode], list[str]]:
    """返回（保留的节点列表，被移除节点的 node_id 列表）。"""
    kept: list[GraphNode] = []
    removed_ids: list[str] = []
    for node in graph.nodes:
        if is_implicit_patient_node(node) or node.source_text in graph.chunk_text:
            kept.append(node)
        else:
            removed_ids.append(node.node_id)
            graph.validation_warnings.append(
                f"[HARD] 节点 {node.node_id}「{node.source_text}」"
                f"不是 chunk_text 的子串，已删除"
            )
    return kept, removed_ids


# ---------------------------------------------------------------------------
# HARD rule: dangling edges
# ---------------------------------------------------------------------------

def _drop_dangling_edges(
    edges: list[GraphEdge],
    valid_node_ids: set[str],
    warnings: list[str],
) -> list[GraphEdge]:
    kept: list[GraphEdge] = []
    for edge in edges:
        if edge.source_node_id not in valid_node_ids:
            warnings.append(
                f"[HARD] 边 {edge.edge_id} source={edge.source_node_id} 节点不存在，已删除"
            )
        elif edge.target_node_id not in valid_node_ids:
            warnings.append(
                f"[HARD] 边 {edge.edge_id} target={edge.target_node_id} 节点不存在，已删除"
            )
        else:
            kept.append(edge)
    return kept


# ---------------------------------------------------------------------------
# HARD rule: edge direction and endpoint node types
# ---------------------------------------------------------------------------

def _drop_endpoint_type_violations(
    edges: list[GraphEdge],
    nodes_by_id: dict[str, GraphNode],
    warnings: list[str],
) -> list[GraphEdge]:
    kept: list[GraphEdge] = []
    for edge in edges:
        constraint = EDGE_ENDPOINT_CONSTRAINTS.get(edge.edge_type)
        if constraint is None:
            warnings.append(
                f"[HARD] 边 {edge.edge_id} type={edge.edge_type} 没有端点约束定义，已删除"
            )
            continue

        allowed_sources, allowed_targets = constraint
        source_type = nodes_by_id[edge.source_node_id].node_type
        target_type = nodes_by_id[edge.target_node_id].node_type
        if source_type not in allowed_sources or target_type not in allowed_targets:
            warnings.append(
                f"[HARD] 边 {edge.edge_id} type={edge.edge_type} 端点类型/方向错误："
                f"{source_type}->{target_type}，已删除"
            )
            continue
        kept.append(edge)
    return kept


# ---------------------------------------------------------------------------
# SOFT: LLM edge validation
# ---------------------------------------------------------------------------

_EDGE_REVIEW_SYSTEM = (
    "你是临床 NLP 专家。请审查以下知识图谱边的推断理由是否合理。"
    "输出 JSON，包含字段 edge_id, verdict（valid/low_confidence/invalid）, verdict_reason（≤60字）。"
)


def _build_edge_review_prompt(edge: GraphEdge, chunk_text: str) -> str:
    return (
        f"Chunk 原文：{chunk_text}\n\n"
        f"边 ID：{edge.edge_id}\n"
        f"边类型：{edge.edge_type}\n"
        f"推断依据片段：{edge.source_text}\n"
        f"推断逻辑：{edge.reasoning}\n"
        f"LLM 自评置信度：{edge.confidence}\n\n"
        "请判断该边的推断是否合理：\n"
        "- valid：推断有充分原文依据，边类型选择正确\n"
        "- low_confidence：推断有一定依据但存在歧义，保留但标记为低置信度\n"
        "- invalid：推断无原文支持或边类型明显错误，应删除\n"
    )


async def _validate_edge_with_llm(
    edge: GraphEdge,
    chunk_text: str,
    validator_cfg: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> EdgeValidationResult:
    prompt = _build_edge_review_prompt(edge, chunk_text)
    async with semaphore:
        try:
            result: EdgeValidationResult = await call_with_instructor_async(
                response_model=EdgeValidationResult,
                messages=[
                    {"role": "system", "content": _EDGE_REVIEW_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                model=validator_cfg.get("model", "gpt-5.5"),
                temperature=validator_cfg.get("temperature", 0.0),
                max_tokens=validator_cfg.get("max_tokens", 256),
                top_p=validator_cfg.get("top_p", 1.0),
                timeout=validator_cfg.get("timeout", 60),
                max_retries=validator_cfg.get("max_retries", 3),
            )
            return result
        except Exception as exc:
            logger.warning("edge validator LLM call failed for %s: %s", edge.edge_id, exc)
            return EdgeValidationResult(
                edge_id=edge.edge_id,
                verdict="low_confidence",
                verdict_reason=f"LLM 调用失败，保守保留: {exc}"[:60],
            )


# ---------------------------------------------------------------------------
# Per-graph soft validation
# ---------------------------------------------------------------------------

async def _soft_validate_edges(
    graph: ChunkGraph,
    validator_cfg: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> None:
    """对需要 LLM 审查的边并发调用，原地修改 graph.edges 和 low_confidence_edge_ids。"""
    threshold: float = validator_cfg.get("confidence_threshold", 0.5)

    # 识别需要 LLM 审查的边
    to_review: list[GraphEdge] = []
    for edge in graph.edges:
        needs_review = (
            edge.confidence < threshold
            or edge.source_text not in graph.chunk_text
        )
        if needs_review:
            to_review.append(edge)

    if not to_review:
        return

    results: list[EdgeValidationResult] = await asyncio.gather(*[
        _validate_edge_with_llm(e, graph.chunk_text, validator_cfg, semaphore)
        for e in to_review
    ])

    # 按结果处理
    invalid_ids: set[str] = set()
    low_conf_ids: set[str] = set()
    for res in results:
        if res.verdict == "invalid":
            invalid_ids.add(res.edge_id)
            graph.validation_warnings.append(
                f"[SOFT-invalid] 边 {res.edge_id}：{res.verdict_reason}"
            )
        elif res.verdict == "low_confidence":
            low_conf_ids.add(res.edge_id)
            graph.validation_warnings.append(
                f"[SOFT-low_conf] 边 {res.edge_id}：{res.verdict_reason}"
            )

    # 移除 invalid 边
    graph.edges = [e for e in graph.edges if e.edge_id not in invalid_ids]
    # 标记 low_confidence 边
    graph.low_confidence_edge_ids.extend(low_conf_ids)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def validate_chunk_graph_async(
    graph: ChunkGraph,
    config_path: Path | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> ChunkGraph:
    """对单个 ChunkGraph 执行全部验证，原地修改并返回。

    Args:
        semaphore: 外部传入时复用（跨 section 全局限速），为 None 时自行创建。
    """
    validator_cfg = _get_validator_cfg(config_path)
    if semaphore is None:
        max_concurrency = validator_cfg.get("max_concurrency", 4)
        semaphore = asyncio.Semaphore(max_concurrency)

    # HARD pre-step: 隐式患者锚点。必须在节点/边硬验证前执行。
    ensure_patient_anchor(graph)

    # HARD: 节点
    valid_nodes, _removed_node_ids = _validate_nodes_hard(graph)
    graph.nodes = valid_nodes

    # HARD: 悬空边
    valid_ids = {n.node_id for n in valid_nodes}
    graph.edges = _drop_dangling_edges(graph.edges, valid_ids, graph.validation_warnings)

    # HARD: 关系方向和端点类型
    nodes_by_id = {n.node_id: n for n in valid_nodes}
    graph.edges = _drop_endpoint_type_violations(
        graph.edges,
        nodes_by_id,
        graph.validation_warnings,
    )

    # SOFT: LLM 审查低置信度/不可定位边
    await _soft_validate_edges(graph, validator_cfg, semaphore)

    return graph


async def validate_chunk_graphs_async(
    graphs: list[ChunkGraph],
    config_path: Path | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> list[ChunkGraph]:
    """批量验证，复用同一组 Semaphore。

    Args:
        semaphore: 外部传入时复用（跨 section 全局限速），为 None 时自行创建。
    """
    validator_cfg = _get_validator_cfg(config_path)
    if semaphore is None:
        max_concurrency = validator_cfg.get("max_concurrency", 4)
        semaphore = asyncio.Semaphore(max_concurrency)

    return await asyncio.gather(*[
        validate_chunk_graph_async(g, config_path, semaphore)
        for g in graphs
    ])
