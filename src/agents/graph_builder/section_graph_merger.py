"""C-D：Section 级 ChunkGraph 合并 → SectionGraph。

合并策略：
  1. 收集同一 section 下所有 ChunkGraph 的节点
  2. 节点合并：
     a. source_text 完全相同 → 直接合并（同一实体）
     b. source_text 包含关系（一方是另一方子串，差值 ≤ MAX_SUBSTRING_DIFF）→ 批量 LLM 共指审查
  3. 节点 ID 从临时格式（n1/n2）重编为全局格式
     {case_id}_sec{si:03d}_nd{ni:04d}
  4. 边：source/target node_id 按映射表重编，跨 chunk 相同边去重
  5. 输出 SectionGraph（Pydantic）+ networkx DiGraph + JSON 文件 + 摘要 txt
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import yaml
from pydantic import BaseModel, ConfigDict
from src.agents.context_chunker.config_loader import resolve_project_path
from src.llm.async_instructor_client import call_with_instructor_async
from src.schemas.graph.clinical_graph import (
    ChunkGraph,
    GraphEdge,
    GraphFrame,
    GraphNode,
    SectionGraph,
    SectionGraphEdge,
    SectionGraphNode,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/agents/graph_builder/graph_builder.yaml")
MAX_SUBSTRING_DIFF = 4   # source_text 长度差 ≤ 此值才触发共指审查


class _CoreferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    are_coreferent: bool
    reason: str


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_merger_cfg(config_path: Path | None = None) -> dict[str, Any]:
    path = resolve_project_path(config_path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"graph_builder config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("section_merger", raw.get("chunk_graph_builder", {}))


# ---------------------------------------------------------------------------
# Step 1: collect nodes & edges from ChunkGraphs
# ---------------------------------------------------------------------------

def _collect(
    graphs: list[ChunkGraph],
) -> tuple[
    dict[str, tuple[GraphNode, str]],   # global_tmp_id → (node, chunk_id)
    list[tuple[GraphEdge, str]],         # (edge, chunk_id)
    list[tuple[GraphFrame, str, str]],    # (frame, chunk_id, graph_prefix)
]:
    """给每个节点和边分配全局临时 ID，防止不同 chunk 内 n1 冲突。"""
    all_nodes: dict[str, tuple[GraphNode, str]] = {}
    all_edges: list[tuple[GraphEdge, str]] = []
    all_frames: list[tuple[GraphFrame, str, str]] = []

    for gi, g in enumerate(graphs):
        prefix = f"g{gi}_"
        for node in g.nodes:
            all_nodes[prefix + node.node_id] = (node, g.chunk_id)
        for edge in g.edges:
            # 重写 edge 的 source/target 为带前缀的 ID
            rewritten = edge.model_copy(update={
                "edge_id": prefix + edge.edge_id,
                "source_node_id": prefix + edge.source_node_id,
                "target_node_id": prefix + edge.target_node_id,
            })
            all_edges.append((rewritten, g.chunk_id))
        for frame in g.frames:
            all_frames.append((frame, g.chunk_id, prefix))

    return all_nodes, all_edges, all_frames


# ---------------------------------------------------------------------------
# Step 2: node merging
# ---------------------------------------------------------------------------

async def _merge_nodes(
    all_nodes: dict[str, tuple[GraphNode, str]],
    merger_cfg: dict[str, Any],
    semaphore: asyncio.Semaphore | None = None,
) -> dict[str, str]:
    """返回 {global_tmp_id → canonical_global_tmp_id} 映射。"""
    # 按 source_text 分组
    by_text: dict[str, list[str]] = defaultdict(list)
    for gtid, (node, _) in all_nodes.items():
        by_text[node.source_text].append(gtid)

    # 完全相同 → 同一组
    merge_map: dict[str, str] = {}
    for gtids in by_text.values():
        canon = gtids[0]
        for gtid in gtids:
            merge_map[gtid] = canon

    # 患者主体是病例级锚点：同一 section 内所有显式/隐式 PATIENT 强制合并。
    patient_gtids = [
        gtid
        for gtid, (node, _) in all_nodes.items()
        if node.node_type == "PATIENT"
    ]
    if patient_gtids:
        explicit = [
            gtid
            for gtid in patient_gtids
            if not all_nodes[gtid][0].implicit
        ]
        patient_canon = explicit[0] if explicit else patient_gtids[0]
        for gtid in patient_gtids:
            merge_map[gtid] = patient_canon

    # 包含关系 → 候选对，LLM 共指审查
    texts = list(by_text.keys())
    coreferent_pairs: list[tuple[str, str]] = []   # (short_text, long_text)
    for i, t1 in enumerate(texts):
        for t2 in texts[i + 1:]:
            short, long = (t1, t2) if len(t1) <= len(t2) else (t2, t1)
            if short in long and len(long) - len(short) <= MAX_SUBSTRING_DIFF:
                coreferent_pairs.append((short, long))

    for short_text, long_text in coreferent_pairs:
        result = await _check_coreference(short_text, long_text, merger_cfg, semaphore)
        if result.are_coreferent:
            # 将 long_text 的所有节点归并到 short_text 的 canonical
            canon = merge_map[by_text[short_text][0]]
            for gtid in by_text[long_text]:
                merge_map[gtid] = canon

    return merge_map


async def _check_coreference(
    text_a: str,
    text_b: str,
    merger_cfg: dict[str, Any],
    semaphore: asyncio.Semaphore | None = None,
) -> _CoreferenceResult:
    prompt = (
        f"在同一份病历中，以下两个实体文本是否指代同一临床概念（共指）？\n"
        f"A: 「{text_a}」\n"
        f"B: 「{text_b}」\n\n"
        "若两者指代完全相同的临床实体（如「气短」和「气短加重」在此语境下均指同一主诉），"
        "则 are_coreferent=true；否则 false。\n"
        "输出 JSON：{\"are_coreferent\": bool, \"reason\": \"一句话原因\"}"
    )
    try:
        if semaphore is not None:
            async with semaphore:
                return await call_with_instructor_async(
                    response_model=_CoreferenceResult,
                    messages=[{"role": "user", "content": prompt}],
                    model=merger_cfg.get("model", "gpt-5.5"),
                    temperature=0.0,
                    max_tokens=128,
                    top_p=1.0,
                    timeout=merger_cfg.get("timeout", 60),
                    max_retries=merger_cfg.get("max_retries", 3),
                )
        return await call_with_instructor_async(
            response_model=_CoreferenceResult,
            messages=[{"role": "user", "content": prompt}],
            model=merger_cfg.get("model", "gpt-5.5"),
            temperature=0.0,
            max_tokens=128,
            top_p=1.0,
            timeout=merger_cfg.get("timeout", 60),
            max_retries=merger_cfg.get("max_retries", 3),
        )
    except Exception as exc:
        logger.warning("coreference check failed for (%s, %s): %s", text_a, text_b, exc)
        return _CoreferenceResult(are_coreferent=False, reason=f"LLM error: {exc}")


# ---------------------------------------------------------------------------
# Step 3: assign global IDs
# ---------------------------------------------------------------------------

def _assign_global_ids(
    all_nodes: dict[str, tuple[GraphNode, str]],
    merge_map: dict[str, str],
    case_id: str,
    section_index: int,
) -> tuple[
    dict[str, SectionGraphNode],   # canonical_gtid → SectionGraphNode
    dict[str, str],                 # global_tmp_id → global_node_id
]:
    # 找出所有 canonical IDs，按 source_text 排序保证稳定性
    canonicals = sorted(
        set(merge_map.values()),
        key=lambda gtid: (
            0 if all_nodes[gtid][0].node_type == "PATIENT" else 1,
            all_nodes[gtid][0].source_text,
        ),
    )

    canon_to_global: dict[str, str] = {}
    section_nodes: dict[str, SectionGraphNode] = {}

    for ni, canon_gtid in enumerate(canonicals, start=1):
        gnode_id = f"{case_id}_sec{section_index:03d}_nd{ni:04d}"
        canon_to_global[canon_gtid] = gnode_id

        # 收集所有归并到此 canonical 的节点的 chunk_id
        merged_tmp_ids = [gtid for gtid, canon in merge_map.items() if canon == canon_gtid]
        chunk_ids = list({all_nodes[gtid][1] for gtid in merged_tmp_ids})

        base_node, _ = all_nodes[canon_gtid]
        section_nodes[gnode_id] = SectionGraphNode(
            node_id=gnode_id,
            node_type=base_node.node_type,
            source_text=base_node.source_text,
            negated=base_node.negated,
            certainty=base_node.certainty,
            implicit=base_node.implicit,
            chunk_ids=chunk_ids,
            merged_from=merged_tmp_ids,
        )

    # 完整映射：每个 global_tmp_id → global_node_id
    full_map = {
        gtid: canon_to_global[canon_gtid]
        for gtid, canon_gtid in merge_map.items()
    }
    return section_nodes, full_map


# ---------------------------------------------------------------------------
# Step 4: merge edges
# ---------------------------------------------------------------------------

def _merge_edges(
    all_edges: list[tuple[GraphEdge, str]],
    node_id_map: dict[str, str],   # global_tmp_id → global_node_id
    case_id: str,
    section_index: int,
    low_conf_ids_per_graph: set[str],
) -> list[SectionGraphEdge]:
    # 去重 key = (global_src, global_tgt, edge_type, negated)
    seen: dict[tuple, SectionGraphEdge] = {}

    for ei, (edge, chunk_id) in enumerate(all_edges, start=1):
        gsrc = node_id_map.get(edge.source_node_id)
        gtgt = node_id_map.get(edge.target_node_id)
        if gsrc is None or gtgt is None:
            # 节点在合并后消失（应该不会）
            continue

        dedup_key = (gsrc, gtgt, edge.edge_type, edge.negated)
        if dedup_key in seen:
            # 追加 chunk_id，取更高置信度
            existing = seen[dedup_key]
            if chunk_id not in existing.chunk_ids:
                seen[dedup_key] = existing.model_copy(update={
                    "chunk_ids": existing.chunk_ids + [chunk_id],
                    "confidence": max(existing.confidence, edge.confidence),
                })
            continue

        global_edge_id = f"{case_id}_sec{section_index:03d}_ed{ei:04d}"
        is_low_conf = edge.edge_id in low_conf_ids_per_graph

        seen[dedup_key] = SectionGraphEdge(
            edge_id=global_edge_id,
            edge_type=edge.edge_type,
            source_node_id=gsrc,
            target_node_id=gtgt,
            negated=edge.negated,
            certainty=edge.certainty,
            context=edge.context,
            source_text=edge.source_text,
            reasoning=edge.reasoning,
            confidence=edge.confidence,
            chunk_ids=[chunk_id],
            low_confidence=is_low_conf,
        )

    return list(seen.values())


def _merge_frames(
    all_frames: list[tuple[GraphFrame, str, str]],
    node_id_map: dict[str, str],
    section_edges: list[SectionGraphEdge],
) -> list[GraphFrame]:
    merged: list[GraphFrame] = []
    for idx, (frame, chunk_id, prefix) in enumerate(all_frames, start=1):
        mapped_nodes = sorted({
            node_id_map[prefix + node_id]
            for node_id in frame.node_ids
            if prefix + node_id in node_id_map
        })
        mapped_edges = sorted({
            edge.edge_id
            for edge in section_edges
            if edge.chunk_ids and chunk_id in edge.chunk_ids
            and (edge.source_node_id in mapped_nodes or edge.target_node_id in mapped_nodes)
        })
        if not mapped_nodes and not mapped_edges:
            continue
        merged.append(GraphFrame(
            frame_id=f"{frame.frame_id}_sec_{idx}",
            frame_type=frame.frame_type,
            source_text=frame.source_text,
            node_ids=mapped_nodes,
            edge_ids=mapped_edges,
            chunk_ids=sorted(set(frame.chunk_ids + [chunk_id])),
        ))
    return merged


# ---------------------------------------------------------------------------
# NetworkX DiGraph builder
# ---------------------------------------------------------------------------

def _build_nx_graph(section_graph: SectionGraph) -> nx.DiGraph:
    G = nx.DiGraph()
    G.graph["graph_id"] = section_graph.graph_id
    G.graph["case_id"] = section_graph.case_id
    G.graph["section_name"] = section_graph.section_name

    for node in section_graph.nodes:
        G.add_node(
            node.node_id,
            node_type=node.node_type,
            source_text=node.source_text,
            negated=node.negated,
            certainty=node.certainty,
            implicit=node.implicit,
        )
    for edge in section_graph.edges:
        G.add_edge(
            edge.source_node_id,
            edge.target_node_id,
            edge_id=edge.edge_id,
            edge_type=edge.edge_type,
            negated=edge.negated,
            confidence=edge.confidence,
            low_confidence=edge.low_confidence,
        )
    return G


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------

def _build_summary(section_graph: SectionGraph) -> str:
    lines = [
        "SectionGraph 摘要",
        f"  graph_id:     {section_graph.graph_id}",
        f"  case_id:      {section_graph.case_id}",
        f"  section:      {section_graph.section_name}（index={section_graph.section_index}）",
        f"  节点数:       {len(section_graph.nodes)}",
        f"  边数:         {len(section_graph.edges)}",
        "",
        "节点列表：",
    ]
    for node in section_graph.nodes:
        neg = " [否定]" if node.negated else ""
        imp = " [隐式]" if node.implicit else ""
        lines.append(f"  {node.node_id}  [{node.node_type}]{neg}{imp}  「{node.source_text}」")
    lines.append("")
    lines.append("边列表：")
    for edge in section_graph.edges:
        lc = " ⚠低置信度" if edge.low_confidence else ""
        lines.append(
            f"  {edge.edge_id}  {edge.source_node_id} --[{edge.edge_type}]--> {edge.target_node_id}"
            f"  conf={edge.confidence:.2f}{lc}"
        )
    if section_graph.frames:
        lines.append("")
        lines.append("上下文框：")
        for frame in section_graph.frames:
            lines.append(
                f"  {frame.frame_id} [{frame.frame_type}] 「{frame.source_text}」 "
                f"nodes={len(frame.node_ids)} edges={len(frame.edge_ids)}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def merge_section_graph_async(
    graphs: list[ChunkGraph],
    section_name: str,
    config_path: Path | None = None,
    output_dir: Path | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> tuple[SectionGraph, nx.DiGraph]:
    """
    将同一 section 下的 ChunkGraph 列表合并为 SectionGraph。

    Args:
        graphs:       已通过 C-4 验证的 ChunkGraph 列表（同一 section）
        section_name: 该 section 的中文名称
        config_path:  可选，graph_builder.yaml 路径
        output_dir:   若提供，则写入 JSON + txt 摘要文件
        semaphore:    外部传入时复用（跨 section 全局限速），为 None 时共指不受限速

    Returns:
        (SectionGraph, nx.DiGraph)
    """
    if not graphs:
        raise ValueError("graphs 不能为空")

    merger_cfg = _load_merger_cfg(config_path)

    case_id = graphs[0].case_id
    section_id = graphs[0].section_id
    section_index = graphs[0].section_index

    # 收集所有 low_confidence edge_id（带前缀后匹配）
    low_conf_edge_ids: set[str] = set()
    for gi, g in enumerate(graphs):
        prefix = f"g{gi}_"
        for eid in g.low_confidence_edge_ids:
            low_conf_edge_ids.add(prefix + eid)

    # C-D 1: 收集
    all_nodes, all_edges, all_frames = _collect(graphs)

    # C-D 2: 节点合并（含 LLM 共指审查）
    merge_map = await _merge_nodes(all_nodes, merger_cfg, semaphore)

    # C-D 3: 分配全局 ID
    section_nodes_dict, node_id_map = _assign_global_ids(
        all_nodes, merge_map, case_id, section_index
    )

    # C-D 4: 边合并
    section_edges = _merge_edges(
        all_edges, node_id_map, case_id, section_index, low_conf_edge_ids
    )
    section_frames = _merge_frames(all_frames, node_id_map, section_edges)

    graph_id = f"{case_id}_sec{section_index:03d}_graph"
    section_graph = SectionGraph(
        graph_id=graph_id,
        case_id=case_id,
        section_id=section_id,
        section_index=section_index,
        section_name=section_name,
        nodes=list(section_nodes_dict.values()),
        edges=section_edges,
        frames=section_frames,
    )

    nx_graph = _build_nx_graph(section_graph)

    # 可选：写文件
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{graph_id}.json"
        json_path.write_text(
            json.dumps(section_graph.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary_path = output_dir / f"{graph_id}_summary.txt"
        summary_path.write_text(_build_summary(section_graph), encoding="utf-8")
        logger.info("SectionGraph saved: %s", json_path)

    return section_graph, nx_graph
