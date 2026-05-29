"""Code-level implicit patient anchor for ChunkGraph.

Chinese medical notes often omit the subject. This module injects one stable
PATIENT node per ChunkGraph and repairs patient-centric relations before hard
edge validation runs.
"""

from __future__ import annotations

import re

from src.schemas.graph.clinical_graph import (
    EDGE_ENDPOINT_CONSTRAINTS,
    ChunkGraph,
    GraphEdge,
    GraphNode,
)

IMPLICIT_PATIENT_NODE_ID = "n0"
IMPLICIT_PATIENT_SOURCE_TEXT = "__PATIENT__"

_PATIENT_SOURCE_EDGE_TYPES: set[str] = {
    "HAS_SEX",
    "HAS_AGE",
    "HAS_OCCUPATION",
    "HAS_BIRTHPLACE",
    "HAS_GENERAL_STATUS",
    "HAS_STATUS_ITEM",
    "HAS_SYMPTOM",
    "SYMPTOM_ONSET",
    "HAS_SIGN",
    "HAS_VITAL_SIGN",
    "HAS_FUNCTIONAL_STATUS",
    "NEGATED_FINDING",
    "DIAGNOSED_WITH",
    "SUSPECTED_DIAGNOSIS",
    "DIAGNOSIS_PENDING",
    "COMORBIDITY",
    "HISTORY_OF",
    "NEGATED_HISTORY_OF",
    "INFECTION_WITH",
    "UNDERWENT_EXAM",
    "TREATED_WITH",
    "UNDERWENT",
    "PARTICIPATED_IN",
    "HAS_ADHERENCE",
    "RECEIVED_VACCINATION",
    "EXPOSED_TO",
    "HAS_HABIT",
}

_TARGET_TYPE_TO_PATIENT_EDGE: dict[str, str] = {
    "SEX": "HAS_SEX",
    "AGE": "HAS_AGE",
    "OCCUPATION": "HAS_OCCUPATION",
    "BIRTHPLACE": "HAS_BIRTHPLACE",
    "GENERAL_HEALTH_STATUS": "HAS_GENERAL_STATUS",
    "GENERAL_STATUS_ITEM": "HAS_STATUS_ITEM",
    "SYMPTOM": "HAS_SYMPTOM",
    "SIGN": "HAS_SIGN",
    "VITAL_SIGN": "HAS_VITAL_SIGN",
    "FUNCTIONAL_STATUS": "HAS_FUNCTIONAL_STATUS",
    "EXAM_TEST": "UNDERWENT_EXAM",
    "DRUG": "TREATED_WITH",
    "PROCEDURE": "UNDERWENT",
    "OXYGEN_THERAPY": "TREATED_WITH",
    "CLINICAL_TRIAL": "PARTICIPATED_IN",
    "VACCINATION": "RECEIVED_VACCINATION",
    "EXPOSURE": "EXPOSED_TO",
    "HABIT": "HAS_HABIT",
    "MEDICATION_ADHERENCE": "HAS_ADHERENCE",
}


def ensure_patient_anchor(graph: ChunkGraph) -> None:
    """Inject and use one PATIENT anchor in-place.

    This does not rely on LLM output. It is deliberately conservative:
    - re-anchor existing patient-centric edges only when their current source
      type violates the endpoint constraint and the target type is compatible;
    - add deterministic patient edges for node types that are inherently about
      the case subject.
    """

    if not graph.nodes:
        return

    patient_id = _ensure_patient_node(graph)
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    _reanchor_patient_edges(graph, patient_id, nodes_by_id)
    _add_missing_patient_edges(graph, patient_id, nodes_by_id)


def is_implicit_patient_node(node: GraphNode) -> bool:
    return (
        node.node_type == "PATIENT"
        and (node.implicit or node.source_text == IMPLICIT_PATIENT_SOURCE_TEXT)
    )


def _ensure_patient_node(graph: ChunkGraph) -> str:
    for node in graph.nodes:
        if node.node_type == "PATIENT":
            return node.node_id

    node_id = _unique_patient_node_id({node.node_id for node in graph.nodes})
    graph.nodes.insert(0, GraphNode(
        node_id=node_id,
        node_type="PATIENT",
        source_text=IMPLICIT_PATIENT_SOURCE_TEXT,
        negated=False,
        certainty="confirmed",
        implicit=True,
    ))
    graph.validation_warnings.append(
        f"[ANCHOR] 已注入隐式 PATIENT 节点 {node_id}，用于承接省略主语关系"
    )
    return node_id


def _unique_patient_node_id(existing_ids: set[str]) -> str:
    if IMPLICIT_PATIENT_NODE_ID not in existing_ids:
        return IMPLICIT_PATIENT_NODE_ID
    idx = 0
    while True:
        candidate = f"n_patient_{idx}"
        if candidate not in existing_ids:
            return candidate
        idx += 1


def _reanchor_patient_edges(
    graph: ChunkGraph,
    patient_id: str,
    nodes_by_id: dict[str, GraphNode],
) -> None:
    patient_node = nodes_by_id[patient_id]
    for edge in graph.edges:
        if edge.edge_type not in _PATIENT_SOURCE_EDGE_TYPES:
            continue
        if edge.source_node_id not in nodes_by_id or edge.target_node_id not in nodes_by_id:
            continue

        source_type = nodes_by_id[edge.source_node_id].node_type
        target_type = nodes_by_id[edge.target_node_id].node_type
        allowed_sources, allowed_targets = EDGE_ENDPOINT_CONSTRAINTS[edge.edge_type]
        if source_type in allowed_sources:
            continue
        if patient_node.node_type not in allowed_sources or target_type not in allowed_targets:
            continue

        old_source = edge.source_node_id
        edge.source_node_id = patient_id
        edge.reasoning = (
            f"代码规则将省略主语的 patient-centric 关系重锚到隐式患者节点；"
            f"原 source={old_source}。{edge.reasoning}"
        )
        graph.validation_warnings.append(
            f"[ANCHOR] 边 {edge.edge_id} type={edge.edge_type} "
            f"由 source={old_source} 重锚到 PATIENT={patient_id}"
        )


def _add_missing_patient_edges(
    graph: ChunkGraph,
    patient_id: str,
    nodes_by_id: dict[str, GraphNode],
) -> None:
    existing = {
        (edge.edge_type, edge.source_node_id, edge.target_node_id)
        for edge in graph.edges
    }
    next_edge_num = _next_edge_number(graph.edges)

    for node in graph.nodes:
        if node.node_id == patient_id:
            continue

        edge_type = _default_patient_edge_type(node, graph.chunk_text, graph.chunk_type)
        if edge_type is None:
            continue
        key = (edge_type, patient_id, node.node_id)
        if key in existing:
            continue
        if not _endpoint_pair_is_allowed(edge_type, "PATIENT", node.node_type):
            continue

        edge_id = f"e_anchor_{next_edge_num}"
        next_edge_num += 1
        graph.edges.append(GraphEdge(
            edge_id=edge_id,
            edge_type=edge_type,  # type: ignore[arg-type]
            source_node_id=patient_id,
            target_node_id=node.node_id,
            negated=node.negated or edge_type.startswith("NEGATED"),
            certainty="excluded" if edge_type.startswith("NEGATED") else node.certainty,
            context=None,
            source_text=_best_source_text(node.source_text, graph.chunk_text),
            reasoning=f"代码规则：{node.node_type} 节点默认归属于病例主体，补充患者锚点关系。",
            confidence=0.9,
        ))
        existing.add(key)
        graph.validation_warnings.append(
            f"[ANCHOR] 已补充 PATIENT --[{edge_type}]--> {node.node_id}"
        )


def _default_patient_edge_type(
    node: GraphNode,
    chunk_text: str,
    chunk_type: str,
) -> str | None:
    if node.negated and node.node_type in {
        "SYMPTOM",
        "SIGN",
        "DISEASE",
        "DISEASE_SUBTYPE",
        "PATHOGEN_OR_INFECTION",
        "IMAGING_FINDING",
        "CARDIAC_FINDING",
        "ECG_FINDING",
        "ULTRASOUND_FINDING",
        "PFT_RESULT",
        "LAB_RESULT",
        "BLOOD_GAS_RESULT",
        "ALLERGY",
    }:
        return "NEGATED_FINDING"

    if node.node_type in {"DISEASE", "DISEASE_SUBTYPE", "PATHOGEN_OR_INFECTION"}:
        return _disease_patient_edge_type(chunk_text, chunk_type)

    if node.node_type == "ALLERGY":
        return "NEGATED_HISTORY_OF" if node.negated or re.search(r"无|否认|未", chunk_text) else "HISTORY_OF"

    return _TARGET_TYPE_TO_PATIENT_EDGE.get(node.node_type)


def _disease_patient_edge_type(chunk_text: str, chunk_type: str) -> str | None:
    if re.search(r"否认|无.*(?:病史|感染|结核|肝炎)|未见", chunk_text):
        return "NEGATED_FINDING"
    if re.search(r"待诊|待排|排查", chunk_text):
        return "DIAGNOSIS_PENDING"
    if re.search(r"考虑|疑诊|可能|倾向", chunk_text):
        return "SUSPECTED_DIAGNOSIS"
    if re.search(r"既往|病史|史[，。；,;]|患.*年|有.*年", chunk_text):
        return "HISTORY_OF"
    if re.search(r"诊断|诊为|确诊|收住", chunk_text) or chunk_type == "diagnosis_entry":
        return "DIAGNOSED_WITH"
    if re.search(r"合并|伴有|同时患有", chunk_text):
        return "COMORBIDITY"
    return None


def _endpoint_pair_is_allowed(edge_type: str, source_type: str, target_type: str) -> bool:
    allowed_sources, allowed_targets = EDGE_ENDPOINT_CONSTRAINTS[edge_type]
    return source_type in allowed_sources and target_type in allowed_targets


def _next_edge_number(edges: list[GraphEdge]) -> int:
    max_num = 0
    for edge in edges:
        match = re.fullmatch(r"e(?:_anchor_)?(\d+)", edge.edge_id)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return max_num + 1


def _best_source_text(node_text: str, chunk_text: str) -> str:
    return chunk_text if len(chunk_text) <= 120 else node_text
