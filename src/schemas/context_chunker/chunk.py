"""SemanticChunk — Step B 输出的最小语义切分单元。

每个 SemanticChunk 对应一个 DiscourseSection 内部的「语义自洽子片段」：
- text 是原文的精确子串
- start_char / end_char 是在 raw_text 中的绝对偏移
- chunk_type 描述这段文字的临床功能类型
- standalone_summary 是供 Step C（构图）使用的一句话摘要
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# chunk_type 枚举
# ---------------------------------------------------------------------------

ChunkType = Literal[
    "temporal_episode",  # 以时间点/时间段为锚的叙事（"8年前…"、"2月前…"）
    "exam_item",         # 一条或一组同类检查 / 检验结果（CT / 肺功能子项 / 实验室组）
    "medication_entry",  # 药物记录（药名 + 剂量 + 用法）
    "procedure_entry",   # 非药物治疗操作记录（吸氧、手术、支气管镜、放疗等措施）
    "exposure_entry",    # 暴露/社会史（职业暴露、吸烟史、家族病史、过敏原等）
    "clinical_state",    # 一般状态、主诉、病程概要（无时序演变、无客观检查）
    "diagnosis_entry",   # 诊断陈述（诊断名 ± 诊断依据）
]


# ---------------------------------------------------------------------------
# 公开 Schema
# ---------------------------------------------------------------------------

class SemanticChunk(BaseModel):
    """一个语义自洽单元，是 DiscourseSection 的子片段。"""

    chunk_id: str = Field(..., description="全局唯一 ID，格式 {case_id}_sec{si:03d}_chk{ci:03d}")
    case_id: str
    section_index: int = Field(..., ge=1, description="所属 DiscourseSection 的 index（1-based）")
    section_id: str
    section_name: str
    chunk_index: int = Field(..., ge=1, description="在本 section 内的编号（1-based）")
    text: str = Field(..., description="原文精确子串，不改写")
    start_char: int = Field(..., ge=0, description="在 raw_text 中的绝对起始偏移")
    end_char: int = Field(..., ge=0, description="在 raw_text 中的绝对结束偏移（不含）")
    chunk_type: ChunkType
    standalone_summary: str = Field(
        ...,
        description="供 Step C（知识图谱构建）使用的一句话临床摘要（≤ 80 字）",
    )

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "case_id": self.case_id,
            "section_index": self.section_index,
            "section_id": self.section_id,
            "section_name": self.section_name,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "chunk_type": self.chunk_type,
            "standalone_summary": self.standalone_summary,
        }


# ---------------------------------------------------------------------------
# LLM 内部输出 Schema（不含 ids / offsets，Post-processing 时再补全）
# ---------------------------------------------------------------------------

class _ChunkItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_index: int = Field(..., description="在本 section 内的编号，从 1 开始")
    text: str = Field(
        ...,
        description=(
            "对应原文段落的完整内容，语义字符（汉字、字母、数字）必须与原文一一对应；"
            "允许在首尾边界裁去孤立的标点或空白，内部标点不得改动"
        ),
    )
    chunk_type: ChunkType = Field(..., description="语义单元类型，从 ChunkType 中选择")
    standalone_summary: str = Field(
        ...,
        description="一句话临床摘要，供后续知识图谱构建使用，不超过 80 字",
    )


class _ChunkSplitterOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunks: list[_ChunkItem] = Field(
        ...,
        description="切分结果列表，按原文出现顺序排列，chunk_index 从 1 递增",
    )
