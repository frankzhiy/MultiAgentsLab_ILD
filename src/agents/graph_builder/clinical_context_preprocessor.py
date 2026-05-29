"""临床上下文预处理器（实体边界 + 状态断言 + ConText 中文实现）。

在 SemanticChunk 进入 LLM 图构建之前，用规则识别三类结构化信息：
  1. ConText cue：真正否定、非否定上下文、稳定状态、未执行等；
  2. protected_entities：医学专有实体的受保护边界，避免同一术语被多粒度拆分；
  3. status_assertions：一般状态 item-value 断言，如「神志 -> 清」。

LLM 在 C-3 阶段必须直接使用这些预标注结果，不重新判断否定，不拆开
protected entity。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from src.agents.context_chunker.config_loader import resolve_project_path
from src.schemas.context_chunker.chunk import SemanticChunk
from src.schemas.graph.clinical_graph import (
    ChunkGraph,
    GraphEdge,
    GraphFrame,
    GraphNode,
    _AnalyzedEntity,
    _AnalyzedRelation,
    _ChunkAnalysis,
)

DEFAULT_TRIGGERS_PATH = Path("configs/agents/graph_builder/negation_triggers.yaml")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NegatedSpan:
    text: str           # 被否定的文字内容（原文子串，无否定触发词本身）
    start_char: int     # 在 chunk.text 内的起始偏移
    end_char: int       # 在 chunk.text 内的结束偏移（不含）
    trigger: str        # 触发该否定的词（如"无"、"否认"）
    pattern: str        # "sequential"（逐项）或 "enumeration"（列举式）
    context_type: str = "negated_finding"
    negated: bool = True


@dataclass(frozen=True)
class ProtectedEntitySpan:
    text: str
    start_char: int
    end_char: int
    node_type: str
    source: str


@dataclass(frozen=True)
class StatusAssertion:
    item_text: str
    value_text: str
    evidence_text: str
    context_text: str | None = None


@dataclass
class AnnotatedChunk:
    chunk: SemanticChunk
    negated_spans: list[NegatedSpan] = field(default_factory=list)
    protected_entities: list[ProtectedEntitySpan] = field(default_factory=list)
    status_assertions: list[StatusAssertion] = field(default_factory=list)

    def format_for_prompt(self) -> str:
        """生成供 LLM 使用的带标注文本，直接嵌入 prompt。"""
        lines = ["原文：", self.chunk.text]
        lines.extend(["", "上下文/否定标注（规则预计算，请直接使用，不要重新判断）："])
        if not self.negated_spans:
            lines.append("  - （无预标注项）")
        for span in self.negated_spans:
            lines.append(
                f"  - [{span.text}] → context_type={span.context_type}, "
                f"negated={span.negated}（触发词：{span.trigger}）"
            )
        lines.extend(["", "受保护医学实体边界（不得拆开或重复用内部词作为平级实体）："])
        if not self.protected_entities:
            lines.append("  - （无）")
        for ent in self.protected_entities:
            lines.append(f"  - [{ent.text}] → node_type={ent.node_type}, source={ent.source}")
        lines.extend(["", "规则化一般状态断言（优先采用 item-value 结构）："])
        if not self.status_assertions:
            lines.append("  - （无）")
        for assertion in self.status_assertions:
            ctx = f"，context={assertion.context_text}" if assertion.context_text else ""
            lines.append(
                f"  - [{assertion.item_text}] -> [{assertion.value_text}]"
                f"（证据：{assertion.evidence_text}{ctx}）"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rule loading
# ---------------------------------------------------------------------------

def _load_rules(path: Path | None = None) -> dict[str, Any]:
    rules_path = path if path is not None else resolve_project_path(str(DEFAULT_TRIGGERS_PATH))
    if not rules_path.exists():
        raise FileNotFoundError(f"negation triggers file not found: {rules_path}")
    with rules_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

_CONTEXT_CUE_PATTERNS: list[tuple[re.Pattern[str], str, bool]] = [
    (re.compile(r"无明显诱因"), "absence_trigger", False),
    (re.compile(r"无明显变化"), "stable_status_value", False),
    (re.compile(r"无特殊不适"), "stable_status_value", False),
    (re.compile(r"未规律[^，。；\n]*(?:口服药物|服药|用药)[^，。；\n]*"), "medication_nonadherence", False),
    (re.compile(r"未予[^，。；\n]*(?:重视|诊疗|治疗|处理|用药|吸氧)[^，。；\n]*"), "care_not_performed", False),
    (re.compile(r"未(?:行|查|复查|完善)[^，。；\n]*"), "exam_or_procedure_not_performed", False),
    (re.compile(r"(?:无法|不能)[^，。；\n]*"), "functional_limitation", False),
    (re.compile(r"(?:考虑|可能|倾向|待排|待诊)[^，。；\n]*"), "uncertainty", False),
    (re.compile(r"(?:既往|曾|目前|入院后|病程中|术后|出院时)"), "temporality", False),
]

_BODY_SITE_TERMS = [
    "双肺下叶基底段",
    "右肺中叶",
    "右肺下叶",
    "左肺舌段",
    "双肺下叶",
    "双下肢",
    "左下肢",
    "右下肢",
    "左膝关节",
    "右膝关节",
    "口唇",
    "甲床",
    "双肺",
    "左肺",
    "右肺",
]

_LEXICON_TERMS: list[tuple[str, str]] = [
    ("沙库巴曲缬沙坦片", "DRUG"),
    ("二甲双胍格列本脲片", "DRUG"),
    ("甲泼尼龙", "DRUG"),
    ("吡非尼酮", "DRUG"),
    ("骨折固定术", "PROCEDURE"),
    ("关节置换术", "PROCEDURE"),
    ("左膝关节置换术", "PROCEDURE"),
    ("支气管镜", "PROCEDURE"),
    ("CAStem细胞注射液输注", "PROCEDURE"),
    ("间质性肺炎", "DISEASE"),
    ("肺部感染", "PATHOGEN_OR_INFECTION"),
    ("新冠病毒感染", "PATHOGEN_OR_INFECTION"),
    ("右束支传导阻滞", "ECG_FINDING"),
    ("中度弥散功能障碍", "PFT_RESULT"),
    ("轻度阻塞性通气功能障碍", "PFT_RESULT"),
]

_PROCEDURE_SUFFIX_RE = re.compile(
    r"[\u4e00-\u9fa5A-Za-z0-9-]{1,12}"
    r"(?:固定术|置换术|切除术|活检术|修复术|成形术|输注|注射|支气管镜)"
)

_DRUG_RE = re.compile(r"[\u4e00-\u9fa5A-Za-z0-9-]{2,24}(?:片|胶囊|注射液|颗粒|滴丸|喷雾剂)")


def _find_context_cues(text: str) -> list[NegatedSpan]:
    spans: list[NegatedSpan] = []
    for pattern, context_type, negated in _CONTEXT_CUE_PATTERNS:
        for match in pattern.finditer(text):
            spans.append(NegatedSpan(
                text=match.group(0),
                start_char=match.start(),
                end_char=match.end(),
                trigger=match.group(0),
                pattern="context_cue",
                context_type=context_type,
                negated=negated,
            ))
    return spans


def _is_inside_context_cue(pos: int, context_spans: list[NegatedSpan]) -> bool:
    return any(sp.start_char <= pos < sp.end_char for sp in context_spans)


def _find_protected_entities(text: str) -> list[ProtectedEntitySpan]:
    spans: list[ProtectedEntitySpan] = []

    for term in sorted(_BODY_SITE_TERMS, key=len, reverse=True):
        for match in re.finditer(re.escape(term), text):
            spans.append(ProtectedEntitySpan(
                text=term,
                start_char=match.start(),
                end_char=match.end(),
                node_type="BODY_SITE",
                source="body_site_lexicon",
            ))

    for term, node_type in sorted(_LEXICON_TERMS, key=lambda x: len(x[0]), reverse=True):
        for match in re.finditer(re.escape(term), text):
            spans.append(ProtectedEntitySpan(
                text=term,
                start_char=match.start(),
                end_char=match.end(),
                node_type=node_type,
                source="medical_lexicon",
            ))

    for pattern, node_type, source in [
        (_DRUG_RE, "DRUG", "drug_suffix"),
        (_PROCEDURE_SUFFIX_RE, "PROCEDURE", "procedure_suffix"),
    ]:
        for match in pattern.finditer(text):
            span_text, start = _strip_leading_non_entity(match.group(0), match.start())
            if not span_text:
                continue
            spans.append(ProtectedEntitySpan(
                text=span_text,
                start_char=start,
                end_char=match.end(),
                node_type=node_type,
                source=source,
            ))

    return _dedupe_protected_entities(spans)


def _strip_leading_non_entity(text: str, start_char: int) -> tuple[str, int]:
    text, start_char = _strip_leading_pattern(text, start_char, r"\d+\s*(?:年|月|天|周|小时)前")
    text, start_char = _strip_leading_pattern(text, start_char, r"(?:既往|曾|目前|入院后|病程中)")
    text, start_char = _strip_leading_pattern(text, start_char, r"(?:口服|静滴|吸入|给予|予以)")
    for site in sorted(_BODY_SITE_TERMS, key=len, reverse=True):
        if text.startswith(site) and len(text) > len(site):
            return text[len(site):], start_char + len(site)
    return text, start_char


def _strip_leading_pattern(text: str, start_char: int, pattern: str) -> tuple[str, int]:
    match = re.match(pattern, text)
    if not match:
        return text, start_char
    return text[match.end():], start_char + match.end()


def _dedupe_protected_entities(
    spans: list[ProtectedEntitySpan],
) -> list[ProtectedEntitySpan]:
    deduped: list[ProtectedEntitySpan] = []
    seen: set[tuple[int, int, str]] = set()
    for span in sorted(spans, key=lambda s: (s.start_char, -(s.end_char - s.start_char))):
        key = (span.start_char, span.end_char, span.node_type)
        if key in seen:
            continue
        if any(
            existing.start_char <= span.start_char
            and span.end_char <= existing.end_char
            and span.node_type == existing.node_type
            for existing in deduped
        ):
            continue
        seen.add(key)
        deduped.append(span)
    return sorted(deduped, key=lambda s: s.start_char)


def _find_status_assertions(text: str) -> list[StatusAssertion]:
    context = "病程中" if "病程中" in text else None
    assertions: list[StatusAssertion] = []

    for item, value in [
        ("神志", "清"),
        ("精神", "可"),
        ("意识", "清"),
        ("饮食", "正常"),
        ("睡眠", "正常"),
        ("大小便", "正常"),
        ("二便", "正常"),
    ]:
        if item in {"饮食", "睡眠"}:
            continue
        pattern = re.compile(rf"({re.escape(item)})\s*({re.escape(value)})")
        for match in pattern.finditer(text):
            assertions.append(StatusAssertion(
                item_text=match.group(1),
                value_text=match.group(2),
                evidence_text=match.group(0),
                context_text=context,
            ))

    for match in re.finditer(r"(饮食睡眠|饮食、睡眠|饮食及睡眠|饮食和睡眠)[^，。；\n]*正常", text):
        evidence = match.group(0)
        assertions.append(StatusAssertion("饮食", "正常", evidence, context))
        assertions.append(StatusAssertion("睡眠", "正常", evidence, context))

    for match in re.finditer(r"(?:饮食睡眠、)?(大小便|二便)[^，。；\n]*正常", text):
        assertions.append(StatusAssertion(match.group(1), "正常", match.group(0), context))

    for match in re.finditer(r"(体重)\s*(无明显变化|稳定|未见明显变化)", text):
        assertions.append(StatusAssertion(
            item_text=match.group(1),
            value_text=match.group(2),
            evidence_text=match.group(0),
            context_text=context,
        ))

    for match in re.finditer(r"(一般健康状况)[:：]?\s*(良好|可|尚可)", text):
        assertions.append(StatusAssertion(
            item_text=match.group(1),
            value_text=match.group(2),
            evidence_text=match.group(0),
            context_text=context,
        ))

    return _dedupe_status_assertions(assertions)


def _dedupe_status_assertions(assertions: list[StatusAssertion]) -> list[StatusAssertion]:
    seen: set[tuple[str, str, str | None]] = set()
    deduped: list[StatusAssertion] = []
    for assertion in assertions:
        key = (assertion.item_text, assertion.value_text, assertion.context_text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(assertion)
    return deduped


def _find_negated_spans(text: str, rules: dict[str, Any]) -> list[NegatedSpan]:
    triggers: list[str] = rules.get("negation_triggers", [])
    right_terminators: list[str] = rules.get("scope_right_terminators", [])
    scope_breakers: list[str] = rules.get("scope_breakers", [])
    list_seps: list[str] = rules.get("list_separators", ["、", "，"])
    sent_boundaries: list[str] = rules.get("sentence_boundaries", ["。", "；", "\n"])

    context_spans = _find_context_cues(text)
    spans: list[NegatedSpan] = list(context_spans)
    enumeration_trigger_ranges: list[tuple[int, int]] = []

    # ------------------------------------------------------------------
    # 句式 B：列举式否定
    # "无A、B、C等不适" 或 "无A、B、C等"
    # ------------------------------------------------------------------
    # 构造右边界模式：等不适 / 等症状 / 等 / 句子边界
    term_pat = "|".join(re.escape(t) for t in right_terminators)
    sep_pat = "[" + "".join(re.escape(s) for s in list_seps) + "]"
    trig_pat = "|".join(re.escape(t) for t in sorted(triggers, key=len, reverse=True))

    # 匹配：触发词 + 第一项 + (分隔符 + 更多项)+ 终止符
    enum_pattern = re.compile(
        rf"({trig_pat})"            # group 1: 触发词
        rf"([^{re.escape(''.join(sent_boundaries))}，。；]+)"  # group 2: 第一项到句界
    )

    for m in enum_pattern.finditer(text):
        if _is_inside_context_cue(m.start(1), context_spans):
            continue
        trigger_word = m.group(1)
        rest = m.group(2)
        rest_start = m.start(2)

        # 检查 rest 内是否有列举分隔符（顿号/逗号）→ 才算列举式
        has_list_sep = bool(re.search(sep_pat, rest))
        # 检查 rest 末尾是否有终止符（"等不适"等）
        has_terminator = bool(re.search(term_pat, rest))

        if not (has_list_sep or has_terminator):
            # 不是列举式，走逐项逻辑（后面处理）
            continue

        enumeration_trigger_ranges.append((m.start(1), m.end(2)))

        # 找终止符位置，截断 rest
        term_m = re.search(term_pat, rest)
        if term_m:
            rest = rest[: term_m.start()]

        # 按分隔符切开
        items = re.split(sep_pat, rest)
        cursor = rest_start
        for item in items:
            stripped = item.strip()
            if not stripped:
                cursor += len(item)
                continue
            # 在原 text 中定位
            idx = text.find(stripped, cursor)
            if idx == -1:
                cursor += len(item)
                continue
            spans.append(NegatedSpan(
                text=stripped,
                start_char=idx,
                end_char=idx + len(stripped),
                trigger=trigger_word,
                pattern="enumeration",
                context_type="negated_finding",
                negated=True,
            ))
            cursor = idx + len(stripped)

    # ------------------------------------------------------------------
    # 句式 A：逐项否定
    # "无发热，无咯血" — 每个触发词单独处理
    # ------------------------------------------------------------------
    # 找所有触发词位置
    for trigger in sorted(triggers, key=len, reverse=True):
        for tm in re.finditer(re.escape(trigger), text):
            trig_start = tm.start()
            trig_end = tm.end()

            if _is_inside_context_cue(trig_start, context_spans):
                continue

            # 如果此位置已被列举式否定覆盖，跳过（避免重复）
            already_covered = any(
                start <= trig_start < end
                for start, end in enumeration_trigger_ranges
            )
            if already_covered:
                continue

            # 向后扫描到句子边界、列举分隔符、截断词或下一个否定触发词。
            scope_end = len(text)
            hard_boundaries = sent_boundaries + list_seps
            for boundary in hard_boundaries:
                bi = text.find(boundary, trig_end)
                if bi != -1 and bi < scope_end:
                    scope_end = bi
            for breaker in scope_breakers:
                bi = text.find(breaker, trig_end)
                if bi != -1 and bi < scope_end:
                    scope_end = bi
            for next_trigger in sorted(triggers, key=len, reverse=True):
                bi = text.find(next_trigger, trig_end)
                if bi != -1 and bi < scope_end:
                    scope_end = bi

            # 提取否定辖域内的内容
            scope_text = text[trig_end:scope_end].strip()
            if not scope_text:
                continue

            # 去掉前导标点
            scope_text = scope_text.lstrip("，、：: ")
            if not scope_text:
                continue

            idx = text.find(scope_text, trig_end)
            if idx == -1:
                continue

            # 避免与已有 spans 重复
            duplicate = any(
                sp.start_char == idx and sp.end_char == idx + len(scope_text)
                for sp in spans
            )
            if duplicate:
                continue

            spans.append(NegatedSpan(
                text=scope_text,
                start_char=idx,
                end_char=idx + len(scope_text),
                trigger=trigger,
                pattern="sequential",
                context_type="negated_finding",
                negated=True,
            ))

    # 去重：相同 (start_char, end_char) 只保留一个
    seen: set[tuple[int, int]] = set()
    deduped: list[NegatedSpan] = []
    for sp in sorted(spans, key=lambda s: s.start_char):
        key = (sp.start_char, sp.end_char)
        if key not in seen:
            seen.add(key)
            deduped.append(sp)

    return deduped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_chunk(
    chunk: SemanticChunk,
    rules: dict[str, Any] | None = None,
) -> AnnotatedChunk:
    """对单个 SemanticChunk 做临床上下文预处理。"""
    if rules is None:
        rules = _load_rules()
    return AnnotatedChunk(
        chunk=chunk,
        negated_spans=_find_negated_spans(chunk.text, rules),
        protected_entities=_find_protected_entities(chunk.text),
        status_assertions=_find_status_assertions(chunk.text),
    )


def annotate_chunks(
    chunks: list[SemanticChunk],
    rules_path: Path | None = None,
) -> list[AnnotatedChunk]:
    """批量标注，复用同一份规则。"""
    rules = _load_rules(rules_path)
    return [annotate_chunk(c, rules) for c in chunks]


# ---------------------------------------------------------------------------
# C-3 normalization hooks
# ---------------------------------------------------------------------------

def normalize_chunk_analysis(ac: AnnotatedChunk, analysis: _ChunkAnalysis) -> _ChunkAnalysis:
    """Normalize C-3a output using deterministic preprocessing.

    The LLM may still over-split medical terms. This function makes protected
    spans and item-value status assertions authoritative before C-3b mapping.
    """

    suppressed = _suppressed_analysis_texts(ac)
    entities = [
        ent for ent in analysis.entities
        if ent.text not in suppressed
    ]
    relations = [
        rel for rel in analysis.relations
        if rel.subject_text not in suppressed and rel.object_text not in suppressed
    ]

    entity_by_text: dict[str, _AnalyzedEntity] = {ent.text: ent for ent in entities}
    for protected in ac.protected_entities:
        if protected.text not in entity_by_text:
            entity_by_text[protected.text] = _AnalyzedEntity(
                text=protected.text,
                description=f"受保护医学实体边界，类型倾向为 {protected.node_type}，不得再拆分。",
                negated=False,
                certainty="confirmed",
            )

    if ac.status_assertions and "患者" in ac.chunk.text and "患者" not in entity_by_text:
        entity_by_text["患者"] = _AnalyzedEntity(
            text="患者",
            description="本段一般状态记录所描述的病例主体。",
            negated=False,
            certainty="confirmed",
        )

    for assertion in ac.status_assertions:
        entity_by_text.setdefault(assertion.item_text, _AnalyzedEntity(
            text=assertion.item_text,
            description="一般状态观察项目。",
            negated=False,
            certainty="confirmed",
        ))
        entity_by_text.setdefault(assertion.value_text, _AnalyzedEntity(
            text=assertion.value_text,
            description="一般状态观察项目的取值。",
            negated=False,
            certainty="confirmed",
        ))
        if "患者" in entity_by_text:
            relations.append(_AnalyzedRelation(
                subject_text="患者",
                object_text=assertion.item_text,
                relation=f"患者有一般状态观察项目：{assertion.item_text}。",
                evidence_text=assertion.evidence_text,
                negated=False,
                certainty="confirmed",
            ))
        relations.append(_AnalyzedRelation(
            subject_text=assertion.item_text,
            object_text=assertion.value_text,
            relation=f"{assertion.item_text}的状态取值为{assertion.value_text}。",
            evidence_text=assertion.evidence_text,
            negated=False,
            certainty="confirmed",
        ))

    for site, proc in _body_site_procedure_pairs(ac.protected_entities):
        relations.append(_AnalyzedRelation(
            subject_text=proc.text,
            object_text=site.text,
            relation=f"{proc.text}涉及的解剖部位是{site.text}。",
            evidence_text=site.text + proc.text,
            negated=False,
            certainty="confirmed",
        ))

    return _ChunkAnalysis(
        overview=analysis.overview,
        entities=list(entity_by_text.values()),
        relations=_dedupe_analysis_relations(relations),
    )


def apply_context_graph_overrides(ac: AnnotatedChunk, graph: ChunkGraph) -> None:
    """Apply deterministic protected-entity and status-assertion graph rules."""

    _apply_protected_entities(ac, graph)
    _apply_status_assertions(ac, graph)


def _suppressed_analysis_texts(ac: AnnotatedChunk) -> set[str]:
    suppressed: set[str] = set()

    for protected in ac.protected_entities:
        for other in _candidate_substrings(ac.chunk.text, protected):
            if other != protected.text:
                suppressed.add(other)

    for assertion in ac.status_assertions:
        suppressed.update(_status_composite_texts(assertion))
    return suppressed


def _candidate_substrings(text: str, protected: ProtectedEntitySpan) -> set[str]:
    candidates: set[str] = set()
    protected_text = protected.text
    for token in ["骨折", "固定术", "置换术", "传导阻滞", "肺炎", "感染", "变化", "明显变化"]:
        if token in protected_text and token != protected_text:
            candidates.add(token)
    for match in re.finditer(r"[\u4e00-\u9fa5A-Za-z0-9-]{1,12}" + re.escape(protected_text), text):
        candidates.add(match.group(0))
    return candidates


def _status_composite_texts(assertion: StatusAssertion) -> set[str]:
    texts = {assertion.evidence_text}
    if assertion.item_text + assertion.value_text == assertion.evidence_text:
        texts.add(assertion.evidence_text)
    if assertion.value_text == "无明显变化":
        texts.update({"明显变化", "变化"})
    if assertion.item_text in {"饮食", "睡眠"}:
        texts.add("饮食睡眠")
    return texts


def _dedupe_analysis_relations(relations: list[_AnalyzedRelation]) -> list[_AnalyzedRelation]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[_AnalyzedRelation] = []
    for rel in relations:
        key = (rel.subject_text, rel.object_text, rel.relation)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rel)
    return deduped


def _apply_protected_entities(ac: AnnotatedChunk, graph: ChunkGraph) -> None:
    if not ac.protected_entities:
        return

    removed_ids: set[str] = set()
    protected_texts = {ent.text for ent in ac.protected_entities}
    for node in graph.nodes:
        if node.source_text in protected_texts:
            protected = next(ent for ent in ac.protected_entities if ent.text == node.source_text)
            node.node_type = protected.node_type  # type: ignore[assignment]
            continue
        if _graph_node_conflicts_with_protected(node, ac.protected_entities):
            removed_ids.add(node.node_id)
            graph.validation_warnings.append(
                f"[BOUNDARY] 节点 {node.node_id}「{node.source_text}」违反受保护医学实体边界，已删除"
            )

    if removed_ids:
        graph.nodes = [node for node in graph.nodes if node.node_id not in removed_ids]
        graph.edges = [
            edge for edge in graph.edges
            if edge.source_node_id not in removed_ids and edge.target_node_id not in removed_ids
        ]

    nodes_by_text = {node.source_text: node for node in graph.nodes}
    for protected in ac.protected_entities:
        if protected.text not in nodes_by_text:
            node = GraphNode(
                node_id=_next_node_id(graph.nodes),
                node_type=protected.node_type,  # type: ignore[arg-type]
                source_text=protected.text,
                negated=False,
                certainty="confirmed",
            )
            graph.nodes.append(node)
            nodes_by_text[node.source_text] = node
            graph.validation_warnings.append(
                f"[BOUNDARY] 已补充受保护医学实体节点 {node.node_id}「{node.source_text}」"
            )

    for site, proc in _body_site_procedure_pairs(ac.protected_entities):
        source = nodes_by_text.get(proc.text)
        target = nodes_by_text.get(site.text)
        if source and target:
            _add_edge_if_missing(
                graph,
                edge_type="LOCALIZES_TO",
                source_node_id=source.node_id,
                target_node_id=target.node_id,
                source_text=site.text + proc.text,
                reasoning="代码规则：部位 + 受保护医学实体，补充操作与部位的定位关系。",
                confidence=0.92,
            )

    for proc in [ent for ent in ac.protected_entities if ent.node_type == "PROCEDURE"]:
        source = nodes_by_text.get(proc.text)
        time_node = _nearest_left_time_node(graph.nodes, proc.start_char, ac.chunk.text)
        if source and time_node:
            _add_edge_if_missing(
                graph,
                edge_type="OCCURS_AT",
                source_node_id=source.node_id,
                target_node_id=time_node.node_id,
                source_text=_bounded_evidence(ac.chunk.text, time_node.source_text, proc.text),
                reasoning="代码规则：时间参照位于受保护操作实体左侧，补充操作发生时间。",
                confidence=0.94,
            )


def _graph_node_conflicts_with_protected(
    node: GraphNode,
    protected_entities: list[ProtectedEntitySpan],
) -> bool:
    for protected in protected_entities:
        if node.source_text == protected.text:
            return False
        if node.source_text in protected.text:
            return True
        if protected.text in node.source_text:
            return True
    return False


def _apply_status_assertions(ac: AnnotatedChunk, graph: ChunkGraph) -> None:
    if not ac.status_assertions:
        return

    status_evidence = {a.evidence_text for a in ac.status_assertions}
    item_texts = {a.item_text for a in ac.status_assertions}
    value_texts = {a.value_text for a in ac.status_assertions}
    remove_ids = {
        node.node_id
        for node in graph.nodes
        if node.node_type == "GENERAL_HEALTH_STATUS"
        and (
            node.source_text not in item_texts | value_texts
            or any(node.source_text in evidence for evidence in status_evidence)
        )
    }
    if remove_ids:
        graph.nodes = [node for node in graph.nodes if node.node_id not in remove_ids]
        graph.edges = [
            edge for edge in graph.edges
            if edge.source_node_id not in remove_ids and edge.target_node_id not in remove_ids
        ]

    nodes_by_key = {(node.node_type, node.source_text): node for node in graph.nodes}
    frame_node_ids: set[str] = set()
    frame_edge_ids: set[str] = set()
    for assertion in ac.status_assertions:
        item_node = _ensure_node(
            graph,
            nodes_by_key,
            node_type="GENERAL_STATUS_ITEM",
            source_text=assertion.item_text,
        )
        value_node = _ensure_node(
            graph,
            nodes_by_key,
            node_type="GENERAL_STATUS_VALUE",
            source_text=assertion.value_text,
        )
        edge = _add_edge_if_missing(
            graph,
            edge_type="STATUS_VALUE_IS",
            source_node_id=item_node.node_id,
            target_node_id=value_node.node_id,
            source_text=assertion.evidence_text,
            reasoning="代码规则：一般状态断言标准化为 item-value 结构。",
            confidence=0.98,
        )
        frame_node_ids.update({item_node.node_id, value_node.node_id})
        if edge:
            frame_edge_ids.add(edge.edge_id)

    context_text = next((a.context_text for a in ac.status_assertions if a.context_text), None)
    if context_text:
        graph.frames.append(GraphFrame(
            frame_id=f"{ac.chunk.chunk_id}_frame_status_1",
            frame_type="temporal_context",
            source_text=context_text,
            node_ids=sorted(frame_node_ids),
            edge_ids=sorted(frame_edge_ids),
            chunk_ids=[ac.chunk.chunk_id],
        ))


def _ensure_node(
    graph: ChunkGraph,
    nodes_by_key: dict[tuple[str, str], GraphNode],
    node_type: str,
    source_text: str,
) -> GraphNode:
    key = (node_type, source_text)
    if key in nodes_by_key:
        return nodes_by_key[key]
    node = GraphNode(
        node_id=_next_node_id(graph.nodes),
        node_type=node_type,  # type: ignore[arg-type]
        source_text=source_text,
        negated=False,
        certainty="confirmed",
    )
    graph.nodes.append(node)
    nodes_by_key[key] = node
    return node


def _add_edge_if_missing(
    graph: ChunkGraph,
    edge_type: str,
    source_node_id: str,
    target_node_id: str,
    source_text: str,
    reasoning: str,
    confidence: float,
) -> GraphEdge | None:
    for edge in graph.edges:
        if (
            edge.edge_type == edge_type
            and edge.source_node_id == source_node_id
            and edge.target_node_id == target_node_id
        ):
            return edge
    edge = GraphEdge(
        edge_id=_next_edge_id(graph.edges),
        edge_type=edge_type,  # type: ignore[arg-type]
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        negated=False,
        certainty="confirmed",
        context=None,
        source_text=source_text,
        reasoning=reasoning,
        confidence=confidence,
    )
    graph.edges.append(edge)
    return edge


def _body_site_procedure_pairs(
    protected_entities: list[ProtectedEntitySpan],
) -> list[tuple[ProtectedEntitySpan, ProtectedEntitySpan]]:
    sites = [ent for ent in protected_entities if ent.node_type == "BODY_SITE"]
    procedures = [ent for ent in protected_entities if ent.node_type == "PROCEDURE"]
    pairs: list[tuple[ProtectedEntitySpan, ProtectedEntitySpan]] = []
    for proc in procedures:
        for site in sites:
            if 0 <= proc.start_char - site.end_char <= 2:
                pairs.append((site, proc))
    return pairs


def _nearest_left_time_node(
    nodes: list[GraphNode],
    right_start: int,
    chunk_text: str,
) -> GraphNode | None:
    candidates: list[tuple[int, GraphNode]] = []
    for node in nodes:
        if node.node_type != "TIME_REF":
            continue
        pos = chunk_text.find(node.source_text)
        if pos == -1 or pos > right_start:
            continue
        if right_start - (pos + len(node.source_text)) <= 12:
            candidates.append((pos, node))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def _bounded_evidence(chunk_text: str, left_text: str, right_text: str) -> str:
    left = chunk_text.find(left_text)
    right = chunk_text.find(right_text, left + len(left_text)) if left != -1 else -1
    if left == -1 or right == -1:
        return right_text
    return chunk_text[left:right + len(right_text)]


def _next_node_id(nodes: list[GraphNode]) -> str:
    max_num = 0
    for node in nodes:
        match = re.fullmatch(r"n(?:_anchor_)?(\d+)", node.node_id)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return f"n_anchor_{max_num + 1}"


def _next_edge_id(edges: list[GraphEdge]) -> str:
    max_num = 0
    for edge in edges:
        match = re.fullmatch(r"e(?:_anchor_)?(\d+)", edge.edge_id)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return f"e_anchor_{max_num + 1}"
