"""ILD 入院记录 / context_chunker / 全 LLM 版 pipeline.

Phases:
  1. Stanza 句切
  2. 句子分类（异步并行）
  3. 全局审查 + 合并相邻同标签 → 篇章 section
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from src.agents.context_chunker.chunk_splitter import split_all_sections_async
from src.agents.context_chunker.config_loader import load_context_chunker_config
from src.agents.context_chunker.sentence_classifier import (
    SubclauseLabel,
    classify_sentences_parallel,
    merge_labels,
    review_labels,
)
from src.agents.context_chunker.sentence_splitter import Sentence, split_sentences
from src.schemas.context_chunker.chunk import SemanticChunk
from src.schemas.context_chunker.discourse_taxonomy import DISCOURSE_TAXONOMY
from src.schemas.context_chunker.raw_text import RawTextInput
from src.utils.id_generator import generate_case_id, generate_text_hash


@dataclass
class DiscourseSection:
    index: int
    section_id: str
    section_name: str
    start_char: int
    end_char: int
    text: str
    sentence_indices: list[int]


@dataclass
class DiscourseSectionedCase:
    case_id: str
    text_hash: str
    raw_text: str
    sentences: list[Sentence] = field(default_factory=list)
    labels: list[SubclauseLabel] = field(default_factory=list)
    sections: list[DiscourseSection] = field(default_factory=list)
    chunks: list[SemanticChunk] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "text_hash": self.text_hash,
            "raw_text": self.raw_text,
            "sentences": [
                {
                    "index": s.index,
                    "start_char": s.start_char,
                    "end_char": s.end_char,
                    "text": s.text,
                }
                for s in self.sentences
            ],
            "labels": [
                {
                    "sentence_index": lab.sentence.index,
                    "initial_section_id": lab.initial_section_id,
                    "initial_reasoning": lab.initial_reasoning,
                    "reviewed_section_id": lab.reviewed_section_id,
                    "reviewed_reasoning": lab.reviewed_reasoning,
                    "changed_in_review": lab.changed_in_review,
                }
                for lab in self.labels
            ],
            "sections": [
                {
                    "index": sec.index,
                    "section_id": sec.section_id,
                    "section_name": sec.section_name,
                    "start_char": sec.start_char,
                    "end_char": sec.end_char,
                    "text": sec.text,
                    "sentence_indices": sec.sentence_indices,
                }
                for sec in self.sections
            ],
            "chunks": [c.to_dict() for c in self.chunks],
        }


def _merge_into_sections(
    raw_text: str,
    labels: list[SubclauseLabel],
) -> list[DiscourseSection]:
    if not labels:
        return []
    sections: list[DiscourseSection] = []
    current_label = labels[0].reviewed_section_id
    current_labels: list[SubclauseLabel] = [labels[0]]

    def flush() -> None:
        sid = current_label
        start = current_labels[0].sentence.start_char
        end = current_labels[-1].sentence.end_char
        name = DISCOURSE_TAXONOMY[sid].zh_name if sid in DISCOURSE_TAXONOMY else sid
        sections.append(DiscourseSection(
            index=len(sections) + 1,
            section_id=sid,
            section_name=name,
            start_char=start,
            end_char=end,
            text=raw_text[start:end],
            sentence_indices=[c.sentence.index for c in current_labels],
        ))

    for lab in labels[1:]:
        if lab.reviewed_section_id == current_label:
            current_labels.append(lab)
        else:
            flush()
            current_label = lab.reviewed_section_id
            current_labels = [lab]
    flush()
    return sections


# --------------------------------------------------------------------------- #
# Main async entry
# --------------------------------------------------------------------------- #

async def section_raw_text_async(
    raw_text: str,
    logger: logging.Logger | None = None,
) -> DiscourseSectionedCase:
    config = load_context_chunker_config()

    validated = RawTextInput(raw_text=raw_text)
    text = validated.raw_text
    case_id = generate_case_id(text)
    text_hash = generate_text_hash(text)

    # Phase 1
    if logger:
        logger.info("  [Phase 1] Stanza 句切…")
    sentences = split_sentences(text)
    if logger:
        logger.info(f"  [Phase 1] ✓ {len(sentences)} 句")

    # Phase 2
    if logger:
        logger.info("  [Phase 2] 句子分类（异步并行）…")
    initial = await classify_sentences_parallel(
        sentences, text, config["sentence_classifier"], logger=logger
    )
    if logger:
        logger.info(f"  [Phase 2] ✓ 完成 {len(initial)} 个句子分类")

    # Phase 3
    if logger:
        logger.info("  [Phase 3] 全局审查…")
    reviewed = await review_labels(
        sentences, text, initial, config["reviewer"], logger=logger
    )
    n_changed = sum(
        1 for s, (init_sid, _), rev in zip(sentences, initial, reviewed, strict=True)
        if rev.section_id != init_sid
    )
    if logger:
        logger.info(f"  [Phase 3] ✓ 审查完成，{n_changed} 个标签被修正")

    labels = merge_labels(sentences, initial, reviewed)
    sections = _merge_into_sections(text, labels)
    if logger:
        logger.info(f"  [Phase 3] ✓ 合并为 {len(sections)} 个篇章 section")

    # Phase 4: 语义子切分
    if logger:
        logger.info("  [Phase 4] 语义子切分（per section LLM）…")
    chunks = await split_all_sections_async(
        sections=sections,
        raw_text=text,
        case_id=case_id,
        config=config["chunk_splitter"],
        logger=logger,
    )
    if logger:
        logger.info(f"  [Phase 4] ✓ 共 {len(chunks)} 个 semantic chunk")

    return DiscourseSectionedCase(
        case_id=case_id,
        text_hash=text_hash,
        raw_text=text,
        sentences=sentences,
        labels=labels,
        sections=sections,
        chunks=chunks,
    )


def section_raw_text(
    raw_text: str,
    logger: logging.Logger | None = None,
) -> DiscourseSectionedCase:
    """同步入口：内部用 asyncio.run 包一层。"""
    return asyncio.run(section_raw_text_async(raw_text, logger=logger))
