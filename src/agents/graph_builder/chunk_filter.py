"""C-1：过滤无效 SemanticChunk，确定哪些 chunk 进入建图流程。

过滤器只做一件事：剔除完全没有临床语义的格式残片，例如单独的逗号、
句号、列表编号（"A1."、"2."）。短文本、人口学信息、一般状态、病史短句、
过渡性诊疗语句都保留给 Step C，由图构建阶段决定其中是否存在可表达的
临床语义。

被排除的 chunk 不进入后续流程，但保留在原列表中，通过 FilterResult 区分。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.schemas.context_chunker.chunk import SemanticChunk

# ---------------------------------------------------------------------------
# FilterResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FilterResult:
    chunk: SemanticChunk
    graphable: bool
    reason: str | None = None   # 排除原因；graphable=True 时为 None


_PUNCT_ONLY_RE = re.compile(r"^[\s,，.。;；:：、/\\|()\[\]{}<>《》“”\"'`~!！?？-]+$")
_LIST_MARKER_RE = re.compile(
    r"^\s*(?:"
    r"[A-Za-z]\d*"
    r"|[A-Za-z]"
    r"|\d+"
    r"|[一二三四五六七八九十]+"
    r")\s*[.)）。、:：]?\s*$"
)


def _is_useless_fragment(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if _PUNCT_ONLY_RE.fullmatch(stripped):
        return True
    return bool(_LIST_MARKER_RE.fullmatch(stripped))


def filter_chunk(chunk: SemanticChunk) -> FilterResult:
    """对单个 chunk 做可建图性判断。"""

    if _is_useless_fragment(chunk.text):
        return FilterResult(
            chunk=chunk,
            graphable=False,
            reason="无临床语义的符号或编号残片",
        )

    return FilterResult(chunk=chunk, graphable=True)


def filter_chunks(chunks: list[SemanticChunk]) -> list[FilterResult]:
    """批量过滤，返回 FilterResult 列表（保留全部 chunk，通过 graphable 字段区分）。"""
    return [filter_chunk(c) for c in chunks]


def get_graphable_chunks(chunks: list[SemanticChunk]) -> list[SemanticChunk]:
    """直接返回可建图的 chunk 列表（过滤掉不可建图的）。"""
    return [r.chunk for r in filter_chunks(chunks) if r.graphable]
