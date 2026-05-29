"""Phase 4: 对每个 DiscourseSection 做语义子切分（Semantic Chunking）。

每个 DiscourseSection 发起一次 LLM 调用，所有 section 并行处理。
LLM 任务：把 section 文本切分为若干「最小语义自洽单元」，
每个单元将独立用于知识图谱构建（Step C）。

后处理流程：
  1. span 对齐：把 LLM 返回的文本片段精确映射到 raw_text 的字符偏移
  2. gap 填充：自动补全 LLM 遗漏的原文片段（保证 100% 覆盖）
  3. 降级保护：LLM 调用失败时，整段作为单个 chunk 兜底
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from src.agents.context_chunker.config_loader import resolve_project_path
from src.llm.async_instructor_client import call_with_instructor_async
from src.schemas.context_chunker.chunk import (
    ChunkType,
    SemanticChunk,
    _ChunkItem,
    _ChunkSplitterOutput,
)
from src.schemas.context_chunker.discourse_taxonomy import DISCOURSE_TAXONOMY
from src.utils.id_generator import generate_chunk_id

# 末尾标点集合（用于宽松匹配时的裁剪）
_TRAILING_PUNCT = frozenset(
    "，。！？；：、…—～「」『』【】〔〕《》〈〉\u201c\u201d\u2018\u2019（）,.!?;:'\")}]"
)

# section 文本长度低于此阈值时直接跳过 LLM，整段作为单 chunk（节省 token）
_SKIP_LLM_THRESHOLD = 30


# ---------------------------------------------------------------------------
# Span alignment
# ---------------------------------------------------------------------------

def _find_text_offset(
    section_text: str,
    chunk_text: str,
    section_start: int,
) -> tuple[int, int] | None:
    """
    在 section_text 中查找 chunk_text，返回 (abs_start, abs_end)。
    按优先级尝试：精确匹配 → 去首尾空白匹配 → 空白归一化匹配 → 去末尾标点宽松匹配。
    全部失败返回 None。
    """
    # 1. 精确匹配
    idx = section_text.find(chunk_text)
    if idx != -1:
        return section_start + idx, section_start + idx + len(chunk_text)

    # 2. 去首尾空白匹配
    stripped = chunk_text.strip()
    if stripped != chunk_text:
        idx = section_text.find(stripped)
        if idx != -1:
            return section_start + idx, section_start + idx + len(stripped)

    # 3. 空白归一化匹配（原文和目标都折叠连续空白为单个空格）
    norm_section = " ".join(section_text.split())
    norm_chunk = " ".join(stripped.split())
    norm_idx = norm_section.find(norm_chunk)
    if norm_idx != -1:
        # 重新在原始 section_text 中查找 stripped（已去掉多余空白的版本）
        idx = section_text.find(stripped)
        if idx != -1:
            return section_start + idx, section_start + idx + len(stripped)

    # 4. 去末尾标点后宽松匹配
    trimmed = norm_chunk.rstrip("".join(_TRAILING_PUNCT))
    if trimmed and trimmed != norm_chunk:
        norm_idx2 = norm_section.find(trimmed)
        if norm_idx2 != -1:
            trimmed_orig = stripped.rstrip("".join(_TRAILING_PUNCT))
            idx = section_text.find(trimmed_orig)
            if idx != -1:
                end = idx + len(trimmed_orig)
                # 顺延包含紧跟的一个标点
                if end < len(section_text) and section_text[end] in _TRAILING_PUNCT:
                    end += 1
                return section_start + idx, section_start + end

    return None


# ---------------------------------------------------------------------------
# Gap filling & chunk assembly
# ---------------------------------------------------------------------------

def _align_and_fill_gaps(
    section_index: int,
    section_id: str,
    section_name: str,
    section_start: int,
    section_end: int,
    raw_text: str,
    llm_items: list[_ChunkItem],
    case_id: str,
    logger: logging.Logger | None,
) -> list[SemanticChunk]:
    """
    1. 对 LLM 输出的每个 chunk item 做 span 对齐。
    2. 按起点排序，去除完全重叠的重复片段。
    3. 填充 LLM 遗漏的 gap，保证完整覆盖 [section_start, section_end)。
    4. 返回排好序的 SemanticChunk 列表（chunk_index 从 1 开始）。
    """
    section_text = raw_text[section_start:section_end]

    # Step 1: 对齐
    aligned: list[tuple[int, int, _ChunkItem]] = []
    for item in llm_items:
        span = _find_text_offset(section_text, item.text, section_start)
        if span is None:
            if logger:
                logger.warning(
                    f"    [Phase4] section {section_id!r}: "
                    f"无法对齐 chunk #{item.chunk_index}: {item.text[:60]!r}"
                )
            continue
        aligned.append((span[0], span[1], item))

    # Step 2: 按起点排序
    aligned.sort(key=lambda x: x[0])

    # Step 3: 去除重叠（保留范围更大的）
    deduped: list[tuple[int, int, _ChunkItem]] = []
    for start, end, item in aligned:
        if deduped and start < deduped[-1][1]:
            prev_start, prev_end, prev_item = deduped[-1]
            if end > prev_end:
                deduped[-1] = (prev_start, end, prev_item)
        else:
            deduped.append((start, end, item))

    # Step 4: 填充 gap，组装最终列表
    # (abs_start, abs_end, text, chunk_type, summary, is_auto_fill)
    final: list[tuple[int, int, str, ChunkType, str]] = []
    cursor = section_start

    for start, end, item in deduped:
        if start > cursor:
            gap_text = raw_text[cursor:start]
            if gap_text.strip():
                final.append((cursor, start, gap_text, "clinical_state", "（自动填充：遗漏片段）"))
        final.append((start, end, raw_text[start:end], item.chunk_type, item.standalone_summary))
        cursor = end

    if cursor < section_end:
        tail_text = raw_text[cursor:section_end]
        if tail_text.strip():
            final.append((cursor, section_end, tail_text, "clinical_state", "（自动填充：末尾片段）"))

    # 生成 SemanticChunk
    chunks: list[SemanticChunk] = []
    for idx, (start, end, text, ctype, summary) in enumerate(final, 1):
        chunks.append(SemanticChunk(
            chunk_id=generate_chunk_id(case_id, section_index, idx),
            case_id=case_id,
            section_index=section_index,
            section_id=section_id,
            section_name=section_name,
            chunk_index=idx,
            text=text,
            start_char=start,
            end_char=end,
            chunk_type=ctype,
            standalone_summary=summary,
        ))

    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def _apply_replacements(template: str, replacements: dict[str, str]) -> str:
    out = template
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out


def _make_single_chunk(
    section_index: int,
    section_id: str,
    section_name: str,
    section_start: int,
    section_end: int,
    raw_text: str,
    case_id: str,
    chunk_type: ChunkType = "clinical_state",
    summary: str | None = None,
) -> list[SemanticChunk]:
    text = raw_text[section_start:section_end]
    return [SemanticChunk(
        chunk_id=generate_chunk_id(case_id, section_index, 1),
        case_id=case_id,
        section_index=section_index,
        section_id=section_id,
        section_name=section_name,
        chunk_index=1,
        text=text,
        start_char=section_start,
        end_char=section_end,
        chunk_type=chunk_type,
        standalone_summary=summary or text.strip()[:80],
    )]


# ---------------------------------------------------------------------------
# Single section splitter（async）
# ---------------------------------------------------------------------------

async def _split_one_section(
    section_index: int,
    section_id: str,
    section_name: str,
    section_start: int,
    section_end: int,
    raw_text: str,
    case_id: str,
    config: dict[str, Any],
    template: str,
    schema_json: str,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger | None,
) -> list[SemanticChunk]:
    """对单个 section 调用 LLM 完成语义子切分，带 semaphore 并发控制。"""

    section_text = raw_text[section_start:section_end]

    # 极短 section 直接跳过 LLM
    if len(section_text.strip()) < _SKIP_LLM_THRESHOLD:
        if logger:
            logger.info(
                f"    [Phase4] [{section_index}] {section_id!r}: "
                f"文本过短（{len(section_text.strip())} 字），跳过 LLM → 1 chunk"
            )
        return _make_single_chunk(
            section_index, section_id, section_name,
            section_start, section_end, raw_text, case_id,
        )

    section_description = (
        DISCOURSE_TAXONOMY[section_id].description
        if section_id in DISCOURSE_TAXONOMY else ""
    )

    prompt = _apply_replacements(template, {
        "{{SECTION_ID}}": section_id,
        "{{SECTION_NAME}}": section_name,
        "{{SECTION_DESCRIPTION}}": section_description,
        "{{SECTION_TEXT}}": section_text,
        "{{OUTPUT_SCHEMA}}": schema_json,
    })

    async with semaphore:
        timeout = config.get("timeout")
        try:
            out = await call_with_instructor_async(
                response_model=_ChunkSplitterOutput,
                messages=[{"role": "user", "content": prompt}],
                model=str(config["model"]),
                temperature=float(config["temperature"]),
                max_tokens=int(config["max_tokens"]),
                top_p=float(config["top_p"]),
                timeout=int(timeout) if timeout is not None else None,
                max_retries=int(config.get("max_retries", 3)),
            )
        except Exception as exc:
            if logger:
                logger.warning(
                    f"    [Phase4] [{section_index}] {section_id!r}: "
                    f"LLM 调用失败，降级为单 chunk: {exc}"
                )
            return _make_single_chunk(
                section_index, section_id, section_name,
                section_start, section_end, raw_text, case_id,
                chunk_type="clinical_state",
                summary="（LLM 降级，整段保留）",
            )

    # --- 对齐 + 填充 ---
    chunks = _align_and_fill_gaps(
        section_index=section_index,
        section_id=section_id,
        section_name=section_name,
        section_start=section_start,
        section_end=section_end,
        raw_text=raw_text,
        llm_items=out.chunks,
        case_id=case_id,
        logger=logger,
    )

    n_auto = sum(1 for ch in chunks if "自动填充" in ch.standalone_summary)
    if logger:
        logger.info(
            f"    [Phase4] [{section_index}] {section_id!r}: "
            f"{len(chunks)} chunks"
            + (f"（含 {n_auto} 个自动填充）" if n_auto else "")
        )

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def split_all_sections_async(
    sections: list,   # list[DiscourseSection] — 避免循环导入，用 Any 规避
    raw_text: str,
    case_id: str,
    config: dict[str, Any],
    logger: logging.Logger | None = None,
) -> list[SemanticChunk]:
    """
    并行对所有 DiscourseSection 做语义子切分。
    返回扁平化的 SemanticChunk 列表，按 section_index → chunk_index 排序。
    """
    template = _read_prompt(resolve_project_path(str(config["prompt_path"])))
    schema_json = json.dumps(
        _ChunkSplitterOutput.model_json_schema(), ensure_ascii=False, indent=2
    )
    max_concurrency = int(config.get("max_concurrency", 4))
    semaphore = asyncio.Semaphore(max_concurrency)

    if logger:
        logger.info(
            f"  [Phase 4] 语义子切分，{len(sections)} 个 section"
            f"，max_concurrency={max_concurrency}"
        )

    tasks = [
        _split_one_section(
            section_index=sec.index,
            section_id=sec.section_id,
            section_name=sec.section_name,
            section_start=sec.start_char,
            section_end=sec.end_char,
            raw_text=raw_text,
            case_id=case_id,
            config=config,
            template=template,
            schema_json=schema_json,
            semaphore=semaphore,
            logger=logger,
        )
        for sec in sections
    ]
    results = await asyncio.gather(*tasks)

    # 展平并按 (section_index, chunk_index) 排序
    all_chunks = [chunk for section_chunks in results for chunk in section_chunks]
    all_chunks.sort(key=lambda c: (c.section_index, c.chunk_index))
    return all_chunks
