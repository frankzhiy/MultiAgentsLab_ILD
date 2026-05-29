from __future__ import annotations

import hashlib


def generate_text_hash(raw_text: str) -> str:
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def generate_case_id(raw_text: str) -> str:
    text_hash = generate_text_hash(raw_text)
    return f"case_{text_hash[:12]}"


def generate_section_id(case_id: str, section_index: int) -> str:
    if section_index < 1:
        raise ValueError("section_index must be greater than or equal to 1")
    return f"{case_id}_sec_{section_index:03d}"


def generate_chunk_id(case_id: str, section_index: int, chunk_index: int) -> str:
    if section_index < 1 or chunk_index < 1:
        raise ValueError("section_index and chunk_index must be >= 1")
    return f"{case_id}_sec{section_index:03d}_chk{chunk_index:03d}"


def generate_node_id(case_id: str, section_index: int, node_index: int) -> str:
    """全局节点 ID，格式：{case_id}_sec{si:03d}_nd{ni:04d}。"""
    if section_index < 1 or node_index < 1:
        raise ValueError("section_index and node_index must be >= 1")
    return f"{case_id}_sec{section_index:03d}_nd{node_index:04d}"


def generate_edge_id(case_id: str, section_index: int, edge_index: int) -> str:
    """全局边 ID，格式：{case_id}_sec{si:03d}_ed{ei:04d}。"""
    if section_index < 1 or edge_index < 1:
        raise ValueError("section_index and edge_index must be >= 1")
    return f"{case_id}_sec{section_index:03d}_ed{edge_index:04d}"


def generate_graph_id(case_id: str, section_index: int) -> str:
    """SectionGraph ID，格式：{case_id}_sec{si:03d}_graph。"""
    if section_index < 1:
        raise ValueError("section_index must be >= 1")
    return f"{case_id}_sec{section_index:03d}_graph"
