"""中文 ILD 入院记录的「篇章 section」分类体系（方案 B / LLM 分类版）。

设计原则：
1. **互斥**：每个子句只能归到一个 section_id。
2. **覆盖**：覆盖 ILD MDT 工作流中典型入院记录的所有篇章惯例。
3. **ILD 导向**：删除月经/婚育等与 ILD 关联弱的类别；把"辅助检查"细化为
   imaging / pulmonary_function / laboratory / pathology 四类——这四类在
   ILD 推理中的权重和后续 facet 抽取需求差异很大。
4. **无 other 兜底**：LLM 必须从这 16 类中选一个；如果实在不确定，
   交给审查阶段处理。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiscourseSectionDefinition:
    zh_name: str
    description: str


DISCOURSE_TAXONOMY: dict[str, DiscourseSectionDefinition] = {
    "general_info": DiscourseSectionDefinition(
        zh_name="一般情况",
        description=(
            "患者人口学与基础健康信息：年龄、性别、职业、籍贯、一般健康状况等。"
            "不含吸烟/饮酒史（那些归 exposure_history）；不含主要症状（那归 chief_complaint）。"
        ),
    ),
    "chief_complaint": DiscourseSectionDefinition(
        zh_name="主诉",
        description=(
            "本次就诊最主要的临床问题，通常是「主因\"…\"入院」或「主诉：…」格式；"
            "由核心症状 + 持续时间构成。"
        ),
    ),
    "present_illness": DiscourseSectionDefinition(
        zh_name="现病史",
        description=(
            "从本次起病到入院之间的连续叙事：症状的起病/演变/缓解/加重、"
            "院外就诊、院外检查（如外院 CT）、院外治疗及反应、转诊原因等。"
            "时间上虽是过去，但都与本次发病相关。"
        ),
    ),
    "past_medical_history": DiscourseSectionDefinition(
        zh_name="既往疾病史",
        description=(
            "与本次发病相对独立的既往疾病：高血压、糖尿病、冠心病、肿瘤、"
            "风湿免疫病、既往肺部感染/结核、既往手术史、外伤史等。"
        ),
    ),
    "medication_history": DiscourseSectionDefinition(
        zh_name="用药史",
        description=(
            "长期或既往使用的药物记录，特别关注 ILD 相关致病药物"
            "（胺碘酮、博来霉素、甲氨蝶呤、来氟米特、呋喃妥因等）和"
            "免疫抑制/激素治疗史。"
        ),
    ),
    "exposure_history": DiscourseSectionDefinition(
        zh_name="暴露史",
        description=(
            "ILD 鉴别诊断的核心：吸烟史、饮酒史、职业粉尘暴露（矽尘/煤尘/石棉）、"
            "有机粉尘（鸟类/羽毛/霉变干草/空调器）、化学品、放射、宠物等。"
        ),
    ),
    "allergy_history": DiscourseSectionDefinition(
        zh_name="过敏史",
        description="药物、食物或其他过敏原；含「否认过敏史」等阴性陈述。",
    ),
    "family_history": DiscourseSectionDefinition(
        zh_name="家族史",
        description=(
            "家族成员的健康情况，尤其是家族性肺纤维化、间质性肺病、风湿免疫病、"
            "肿瘤等可遗传或聚集发病的疾病。"
        ),
    ),
    "physical_exam": DiscourseSectionDefinition(
        zh_name="体格检查",
        description=(
            "入院查体的客观体征：生命体征、一般状况、心肺听诊（特别是 velcro 啰音）、"
            "杵状指、皮疹、关节体征等。"
        ),
    ),
    "imaging_findings": DiscourseSectionDefinition(
        zh_name="影像学检查",
        description=(
            "**入院后完善的**胸部 X 线、CT、HRCT、超声等影像学描述与结论。"
            "外院 CT 描述归 present_illness。"
        ),
    ),
    "pulmonary_function": DiscourseSectionDefinition(
        zh_name="肺功能",
        description=(
            "肺功能检查：通气功能、换气功能（DLCO）、脉冲振荡、6 分钟步行试验、"
            "血气分析。ILD 严重度与随访的核心指标。"
        ),
    ),
    "laboratory_findings": DiscourseSectionDefinition(
        zh_name="实验室检查",
        description=(
            "血常规、生化、肝肾功、炎症指标（CRP/ESR）、凝血、D-二聚体、"
            "自身抗体、感染指标（T-SPOT、病原学）等化验结果。"
        ),
    ),
    "pathology_findings": DiscourseSectionDefinition(
        zh_name="病理与 BALF",
        description=(
            "肺活检（TBLB/TBLC/外科活检）的病理描述与诊断、"
            "BALF 细胞计数与分类、微生物培养等组织/细胞学证据。"
        ),
    ),
    "initial_diagnosis": DiscourseSectionDefinition(
        zh_name="初步诊断",
        description="入院诊断、初步诊断或门诊诊断列表（明确给出的诊断名）。",
    ),
    "treatment_plan": DiscourseSectionDefinition(
        zh_name="治疗与管理计划",
        description="下一步的治疗、检查、会诊、随访计划，以及尚未执行的诊疗安排。",
    ),
    "progress_note": DiscourseSectionDefinition(
        zh_name="病程记录",
        description=(
            "入院后的住院过程记录：病情变化、治疗反应、复查结果、"
            "「病程中，患者神志清、精神可…」等一般病程描述。"
        ),
    ),
}


VALID_SECTION_IDS: frozenset[str] = frozenset(DISCOURSE_TAXONOMY.keys())


def is_valid_section_id(section_id: str) -> bool:
    return section_id in VALID_SECTION_IDS


def format_taxonomy_for_prompt() -> str:
    """生成 LLM prompt 用的 taxonomy 描述。"""
    lines: list[str] = []
    for sid, definition in DISCOURSE_TAXONOMY.items():
        lines.append(f"- `{sid}`（{definition.zh_name}）")
        lines.append(f"  {definition.description}")
    return "\n".join(lines)
