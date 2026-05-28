"""中文句子切分。

底层使用 Stanza 中文 tokenize（gsdsimp 模型），并在其结果上做两步后处理：

1. **强制按换行切分**：中文病历经常用换行而非句号分隔不同篇章块
   （例如「……骨折固定术\\n患者于 8 年前……」），Stanza 不会在换行处断句，
   我们需要把跨换行的句子拆开。
2. **过滤空白句**：去掉只有空白字符的"句子"。

返回的每个 `Sentence` 都带有相对于原文的字符级 offset，便于下游做 span 对齐。
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

_PIPELINE: Any | None = None
_PIPELINE_LOCK = Lock()


@dataclass(frozen=True)
class Sentence:
    index: int                # 1-based 全局编号 S{index}
    start_char: int
    end_char: int
    text: str                 # 原文片段（含首尾原始字符）


def _get_pipeline() -> Any:
    global _PIPELINE
    if _PIPELINE is None:
        with _PIPELINE_LOCK:
            if _PIPELINE is None:
                import stanza
                _PIPELINE = stanza.Pipeline(
                    lang="zh-hans",
                    processors="tokenize",
                    use_gpu=False,
                    verbose=False,
                    download_method=None,  # 假定模型已经下载
                )
    return _PIPELINE


def split_sentences(raw_text: str) -> list[Sentence]:
    """把原文切成句子列表，并保留每个句子在原文中的 [start_char, end_char)。"""
    if not raw_text.strip():
        return []

    doc = _get_pipeline()(raw_text)

    spans: list[tuple[int, int]] = []
    for sent in doc.sentences:
        if not sent.tokens:
            continue
        start = sent.tokens[0].start_char
        end = sent.tokens[-1].end_char
        spans.append((start, end))

    # 后处理：在换行处强制再切一刀。
    refined: list[tuple[int, int]] = []
    for start, end in spans:
        segment = raw_text[start:end]
        cursor = start
        for line in segment.splitlines(keepends=True):
            stripped_len = len(line.rstrip("\r\n"))
            if stripped_len > 0:
                refined.append((cursor, cursor + stripped_len))
            cursor += len(line)

    # 过滤掉只剩空白的片段，并按起点排序去重。
    cleaned: list[Sentence] = []
    seen: set[tuple[int, int]] = set()
    for start, end in sorted(refined):
        if (start, end) in seen:
            continue
        seen.add((start, end))
        text = raw_text[start:end].strip()
        if not text:
            continue
        cleaned.append(
            Sentence(
                index=len(cleaned) + 1,
                start_char=start,
                end_char=end,
                text=raw_text[start:end],
            )
        )

    return cleaned
