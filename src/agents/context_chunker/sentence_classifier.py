"""Phase 2 + Phase 3: 句子分类（异步并行）和全局审查。"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.agents.context_chunker.config_loader import resolve_project_path
from src.agents.context_chunker.sentence_splitter import Sentence
from src.llm.async_instructor_client import call_with_instructor_async
from src.schemas.context_chunker.discourse_taxonomy import (
    DISCOURSE_TAXONOMY,
    format_taxonomy_for_prompt,
)

# 让 LLM 只能从 taxonomy 的 id 中选——Literal 类型由 enum 动态生成。
_SectionIdLiteral = Literal[  # type: ignore[misc]
    "general_info",
    "chief_complaint",
    "present_illness",
    "past_medical_history",
    "medication_history",
    "exposure_history",
    "allergy_history",
    "family_history",
    "physical_exam",
    "imaging_findings",
    "pulmonary_function",
    "laboratory_findings",
    "pathology_findings",
    "initial_diagnosis",
    "treatment_plan",
    "progress_note",
]


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class _ClassificationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section_id: _SectionIdLiteral
    reasoning: str = Field(default="")


class _ReviewItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subclause_index: int
    section_id: _SectionIdLiteral
    reasoning: str = Field(default="")


class _ReviewOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[_ReviewItem]


@dataclass(frozen=True)
class SubclauseLabel:
    sentence: Sentence
    initial_section_id: str
    initial_reasoning: str
    reviewed_section_id: str
    reviewed_reasoning: str  # 仅在审查阶段被修改时填写

    @property
    def changed_in_review(self) -> bool:
        return self.initial_section_id != self.reviewed_section_id


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _apply_replacements(template: str, replacements: dict[str, str]) -> str:
    out = template
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out


# --------------------------------------------------------------------------- #
# Phase 3: 单子句分类（异步并行）
# --------------------------------------------------------------------------- #

async def _classify_one(
    sentence: Sentence,
    raw_text: str,
    classifier_cfg: dict[str, Any],
    template: str,
    schema_json: str,
    semaphore: asyncio.Semaphore,
) -> _ClassificationOutput:
    async with semaphore:
        prompt = _apply_replacements(template, {
            "{{TAXONOMY}}": format_taxonomy_for_prompt(),
            "{{RAW_TEXT}}": raw_text,
            "{{START}}": str(sentence.start_char),
            "{{END}}": str(sentence.end_char),
            "{{SUBCLAUSE_TEXT}}": sentence.text,
            "{{OUTPUT_SCHEMA}}": schema_json,
        })
        timeout = classifier_cfg.get("timeout")
        return await call_with_instructor_async(
            response_model=_ClassificationOutput,
            messages=[{"role": "user", "content": prompt}],
            model=str(classifier_cfg["model"]),
            temperature=float(classifier_cfg["temperature"]),
            max_tokens=int(classifier_cfg["max_tokens"]),
            top_p=float(classifier_cfg["top_p"]),
            timeout=int(timeout) if timeout is not None else None,
            max_retries=int(classifier_cfg.get("max_retries", 3)),
        )


async def classify_sentences_parallel(
    sentences: list[Sentence],
    raw_text: str,
    classifier_cfg: dict[str, Any],
    logger: logging.Logger | None = None,
) -> list[tuple[str, str]]:
    """对每个句子并行调 LLM 做分类。返回 [(section_id, reasoning), ...] 与句子一一对应。"""
    template = _read_prompt(resolve_project_path(str(classifier_cfg["prompt_path"])))
    schema_json = json.dumps(
        _ClassificationOutput.model_json_schema(), ensure_ascii=False, indent=2
    )
    max_concurrency = int(classifier_cfg.get("max_concurrency", 8))
    semaphore = asyncio.Semaphore(max_concurrency)

    if logger:
        logger.info(
            f"    并发分类 {len(sentences)} 个句子"
            f"，max_concurrency={max_concurrency}"
        )

    tasks = [
        _classify_one(s, raw_text, classifier_cfg, template, schema_json, semaphore)
        for s in sentences
    ]
    results = await asyncio.gather(*tasks)
    return [(r.section_id, r.reasoning) for r in results]


# --------------------------------------------------------------------------- #
# Phase 4: 全局审查
# --------------------------------------------------------------------------- #

def _format_initial_labels(
    sentences: list[Sentence],
    initial: list[tuple[str, str]],
) -> str:
    lines = []
    for s, (sid, reason) in zip(sentences, initial, strict=True):
        reason_brief = (reason or "").replace("\n", " ")[:120]
        lines.append(
            f"- S{s.index} [{s.start_char}:{s.end_char}] "
            f"label=`{sid}`  text={s.text!r}  reason={reason_brief}"
        )
    return "\n".join(lines)


async def review_labels(
    sentences: list[Sentence],
    raw_text: str,
    initial_results: list[tuple[str, str]],
    reviewer_cfg: dict[str, Any],
    logger: logging.Logger | None = None,
) -> list[_ReviewItem]:
    """全局审查；返回每个句子的（可能被修改的）标签。"""
    template = _read_prompt(resolve_project_path(str(reviewer_cfg["prompt_path"])))
    schema_json = json.dumps(
        _ReviewOutput.model_json_schema(), ensure_ascii=False, indent=2
    )
    prompt = _apply_replacements(template, {
        "{{TAXONOMY}}": format_taxonomy_for_prompt(),
        "{{RAW_TEXT}}": raw_text,
        "{{INITIAL_LABELS}}": _format_initial_labels(sentences, initial_results),
        "{{OUTPUT_SCHEMA}}": schema_json,
    })

    if logger:
        logger.info(f"    全局审查 {len(sentences)} 个句子…")

    timeout = reviewer_cfg.get("timeout")
    out = await call_with_instructor_async(
        response_model=_ReviewOutput,
        messages=[{"role": "user", "content": prompt}],
        model=str(reviewer_cfg["model"]),
        temperature=float(reviewer_cfg["temperature"]),
        max_tokens=int(reviewer_cfg["max_tokens"]),
        top_p=float(reviewer_cfg["top_p"]),
        timeout=int(timeout) if timeout is not None else None,
        max_retries=int(reviewer_cfg.get("max_retries", 3)),
    )

    by_index = {item.subclause_index: item for item in out.items}
    return [by_index.get(s.index, _ReviewItem(
        subclause_index=s.index, section_id=initial_results[i][0], reasoning=""
    )) for i, s in enumerate(sentences)]


# --------------------------------------------------------------------------- #
# 合并 initial + reviewed → SubclauseLabel
# --------------------------------------------------------------------------- #

def merge_labels(
    sentences: list[Sentence],
    initial: list[tuple[str, str]],
    reviewed: list[_ReviewItem],
) -> list[SubclauseLabel]:
    out: list[SubclauseLabel] = []
    for s, (init_sid, init_reason), rev in zip(sentences, initial, reviewed, strict=True):
        out.append(SubclauseLabel(
            sentence=s,
            initial_section_id=init_sid,
            initial_reasoning=init_reason,
            reviewed_section_id=rev.section_id,
            reviewed_reasoning=rev.reasoning,
        ))
    return out


__all__ = [
    "SubclauseLabel",
    "classify_sentences_parallel",
    "review_labels",
    "merge_labels",
    "DISCOURSE_TAXONOMY",
]
