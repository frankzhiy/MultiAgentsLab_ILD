"""临床知识图谱 Schema（Step C 输出）。

层次：
  ChunkGraph   — 单个 SemanticChunk 产出的局部子图
  SectionGraph — 同一 DiscourseSection 下所有 ChunkGraph 合并后的图

节点设计原则：
  - source_text 必须是 chunk 原文中可精确定位的连续子串
  - 不创造原文中不存在的文字作为节点

边设计原则：
  - source_text 记录推断该关系的原文片段（不要求精确匹配，但须可定位）
  - reasoning 由 LLM 填写，解释 source_text → edge_type 的推断逻辑
  - confidence 由 LLM 自评；低置信度边进入二次 LLM 审查
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# NodeType
# ---------------------------------------------------------------------------

NodeType = Literal[
    # 主体
    "PATIENT",               # 患者本体；可由代码注入为隐式病例主体锚点

    # 人口学 / 一般信息
    "SEX",                   # 性别（男、女、男性、女性）
    "AGE",                   # 年龄（77岁、老年）
    "OCCUPATION",            # 职业（退休教师、面粉加工工作）
    "BIRTHPLACE",            # 出生地 / 籍贯
    "GENERAL_HEALTH_STATUS", # 一般健康状态（良好、神志清、精神可、二便正常等）
    "GENERAL_STATUS_ITEM",   # 一般状态项目（神志、精神、饮食、睡眠、大小便、体重）
    "GENERAL_STATUS_VALUE",  # 一般状态取值（清、可、正常、无明显变化）

    # 临床表现
    "SYMPTOM",               # 症状（咳嗽、胸闷气短、乏力）
    "SIGN",                  # 体征（杵状指、velcro啰音、口唇发绀）
    "VITAL_SIGN",            # 生命体征 / 床旁指标（体温、脉搏、血压、SpO2）
    "QUALIFIER",             # 修饰属性（量不多、灰色、夜间为著）
    "FUNCTIONAL_STATUS",     # 功能状态 / 活动耐量（生活工作无明显受限、活动爬三层楼）

    # 诊断
    "DISEASE",               # 疾病 / 诊断名（间质性肺炎、UIP、过敏性肺炎）
    "DISEASE_SUBTYPE",       # 疾病亚型（IPAF、IIPs 亚型）
    "DIAGNOSIS_ASSERTION",   # 诊断陈述（诊断、考虑诊断、待诊、收住诊断）
    "DIAGNOSIS_BASIS",       # 诊断依据陈述（"符合 UIP 模式"）
    "ALLERGY",               # 过敏史 / 过敏原陈述
    "FAMILY_HISTORY_ITEM",   # 家族史项目
    "PATHOGEN_OR_INFECTION", # 病原体/感染事件（新冠病毒感染、肺部感染、细菌性肺炎）

    # 时间
    "TIME_REF",              # 时间参照点（8年前、2024年6月11日、入院时）
    "DURATION",              # 持续时长（3个月、长期）
    "FREQUENCY",             # 频率（1-2次/天、QD、BID、每日2次）
    "CLINICAL_EVENT",        # 临床事件（入院、出院、就诊、再次加重）
    "ENCOUNTER_LOCATION",    # 就诊地点 / 医疗机构（当地医院、我院急诊）

    # 病因 / 诱因
    "TRIGGER",               # 诱发因素（新冠病毒感染、无明显诱因）
    "EXPOSURE",              # 暴露史实体（粉尘、鸟类接触、吸烟20年）
    "EXPOSURE_AMOUNT",       # 暴露量（每日1包、20支/日）
    "EXPOSURE_DURATION",     # 暴露持续时间（吸烟30年、戒烟2年余）
    "HABIT",                 # 生活习惯（吸烟、饮酒、戒烟）
    "PATHOGEN",              # 病原体（结核杆菌、烟曲霉）
    "VACCINATION",           # 疫苗接种事件（接种新冠疫情疫苗）

    # 药物 / 治疗
    "DRUG",                  # 药物名（甲泼尼龙、吡非尼酮、沙库巴曲缬沙坦）
    "DRUG_DOSE",             # 药物剂量（40mg、2片/次、0.1g/次）
    "DRUG_ROUTE",            # 给药途径（口服、吸入、静滴）
    "DRUG_FREQUENCY",        # 药物频率（1次/日、每日2次、QD、BID）
    "PROCEDURE",             # 手术 / 操作（支气管镜、VATS、骨折固定术）
    "OXYGEN_THERAPY",        # 氧疗（吸氧、家庭氧疗）
    "CLINICAL_TRIAL",        # 临床试验 / 研究项目
    "CONSULTATION",          # 会诊 / MDT 讨论
    "MEDICATION_ADHERENCE",  # 用药依从性（未规律口服药物、持续口服中药）
    "TREATMENT_PLAN",        # 治疗方案描述（抗感染治疗、激素冲击）
    "TREATMENT_RESPONSE",    # 疗效结果（改善不明显、症状好转）

    # 检查结果
    "EXAM_TEST",             # 检查项目（胸部CT、HRCT、CTPA、心电图、彩超）
    "LAB_TEST",              # 化验项目名（CRP、KL-6、血沉）
    "LAB_RESULT",            # 化验结果值（CRP 41.8mg/L）
    "BLOOD_GAS_METRIC",      # 血气指标（PH、PCO2、PO2、SpO2、OI）
    "BLOOD_GAS_RESULT",      # 血气结果值
    "AUTOANTIBODY",          # 自身抗体（ANA、anti-MDA5、anti-Jo-1）
    "IMAGING_FINDING",       # 影像所见（蜂窝影、磨玻璃影、网格影、间质增粗）
    "IMAGING_PATTERN",       # 影像模式（UIP pattern、NSIP pattern）
    "PFT_METRIC",            # 肺功能指标名（FVC、DLCO、TLC）
    "PFT_RESULT",            # 肺功能结果（FVC 65%pred、中度弥散障碍）
    "CARDIAC_FINDING",       # 心超 / 心脏相关所见（左房增大、瓣膜反流）
    "ECG_FINDING",           # 心电图所见（窦性心律、右束支传导阻滞）
    "ULTRASOUND_FINDING",    # 普通超声/彩超所见（胆囊附壁结晶、深静脉未见血栓）
    "PATHOLOGY_FINDING",     # 病理所见（UIP组织学、机化性肺炎）

    # 解剖 / 程度
    "BODY_SITE",             # 解剖部位（右肺中叶、双肺下叶基底段）
    "SEVERITY",              # 严重程度（轻度、中度、重度）
]


# ---------------------------------------------------------------------------
# EdgeType
# ---------------------------------------------------------------------------

EdgeType = Literal[
    # 主体 / 人口学 / 一般状态
    "HAS_SEX",                # 患者性别         PATIENT → SEX
    "HAS_AGE",                # 患者年龄         PATIENT → AGE
    "HAS_OCCUPATION",         # 患者职业         PATIENT → OCCUPATION
    "HAS_BIRTHPLACE",         # 出生地           PATIENT → BIRTHPLACE
    "HAS_GENERAL_STATUS",     # 一般状态         PATIENT → GENERAL_HEALTH_STATUS
    "HAS_STATUS_ITEM",        # 一般状态项目      PATIENT → GENERAL_STATUS_ITEM
    "STATUS_VALUE_IS",        # 状态项目取值      GENERAL_STATUS_ITEM → GENERAL_STATUS_VALUE

    # 症状 / 体征关系
    "HAS_SYMPTOM",            # 患者有症状       PATIENT → SYMPTOM
    "SYMPTOM_ONSET",          # 出现症状         → SYMPTOM
    "HAS_SIGN",               # 有体征           → SIGN
    "HAS_VITAL_SIGN",         # 生命体征         PATIENT → VITAL_SIGN
    "HAS_FUNCTIONAL_STATUS",  # 功能状态         PATIENT → FUNCTIONAL_STATUS
    "NEGATED_FINDING",        # 否认症状/体征/异常 → SYMPTOM / SIGN / DISEASE [negated=True]
    "SYMPTOM_WORSENS_WITH",   # 症状加重因素      SYMPTOM → TRIGGER/CONDITION
    "SYMPTOM_RELIEVES_WITH",  # 症状缓解因素      SYMPTOM → TRIGGER/CONDITION
    "WORSENS_AT_OR_FOR",      # 症状在某时间/持续时长加重 SYMPTOM → TIME_REF/DURATION
    "WORSENS_WITH",           # 症状因某诱因/情境加重 SYMPTOM → TRIGGER/QUALIFIER
    "RELIEVES_WITH",          # 症状因治疗/因素缓解 SYMPTOM → DRUG/TREATMENT_PLAN/OXYGEN_THERAPY
    "ACCOMPANIED_BY",         # 伴随             SYMPTOM → SYMPTOM
    "QUALIFIED_BY",           # 被修饰           SYMPTOM/SIGN/FINDING → QUALIFIER
    "QUANTITY_OF",            # 量修饰           QUALIFIER → SYMPTOM/SIGN
    "COLOR_OF",               # 颜色修饰         QUALIFIER → SYMPTOM/SIGN
    "TEXTURE_OF",             # 性状修饰         QUALIFIER → SYMPTOM/SIGN
    "TIMING_OF",              # 时间特征修饰      QUALIFIER → SYMPTOM/SIGN
    "LOCALIZES_TO",           # 定位于           SYMPTOM/FINDING → BODY_SITE
    "MEASURED_VALUE_IS",      # 生命体征/检查指标的数值 VITAL_SIGN/PFT_METRIC/LAB_TEST → RESULT
    "SEVERITY_IS",            # 程度             SYMPTOM/FINDING/PFT → SEVERITY

    # 时间关系
    "OCCURS_AT",              # 发生时间点        EVENT → TIME_REF
    "LASTS_FOR",              # 持续时长         EVENT → DURATION
    "TEMPORAL_PRECEDES",      # 时序先后         EVENT → EVENT
    "TRIGGERED_BY",           # 诱发于           SYMPTOM/DISEASE → TRIGGER
    "AFTER_EVENT",            # 某事件之后发生   SYMPTOM/DISEASE/EVENT → CLINICAL_EVENT/VACCINATION/PROCEDURE

    # 诊断关系
    "DIAGNOSED_WITH",         # 明确诊断         PATIENT → DISEASE
    "SUSPECTED_DIAGNOSIS",    # 疑诊            PATIENT → DISEASE [certainty=suspected]
    "DIAGNOSIS_PENDING",      # 待诊             PATIENT → DISEASE
    "DIAGNOSIS_ASSERTS",      # 诊断陈述指向疾病  DIAGNOSIS_ASSERTION → DISEASE
    "DIFFERENTIAL_FOR",       # 鉴别诊断         DISEASE → DISEASE
    "SUPPORTED_BY",           # 诊断依据         DISEASE → FINDING/LAB_RESULT/PFT_RESULT
    "CONTRADICTS",            # 不支持           DISEASE → FINDING
    "DIAGNOSIS_CHANGED_TO",   # 诊断更新         DISEASE → DISEASE

    # 合并症 / 病史关系
    "COMORBIDITY",            # 合并症           PATIENT → DISEASE
    "HISTORY_OF",             # 既往史           PATIENT → DISEASE/PROCEDURE
    "NEGATED_HISTORY_OF",     # 否认既往史       PATIENT → DISEASE/PROCEDURE/ALLERGY
    "CAUSED_BY",              # 继发于           DISEASE → DISEASE/EXPOSURE
    "INFECTION_WITH",         # 感染/病原相关    PATIENT/DISEASE → PATHOGEN_OR_INFECTION/PATHOGEN

    # 检查关系
    "UNDERWENT_EXAM",         # 接受/完善检查     PATIENT → EXAM_TEST
    "HAS_RESULT",             # 检查项目→结果    LAB_TEST → LAB_RESULT
    "HAS_FINDING",            # 有所见            EXAM_TEST/BODY_SITE → FINDING
    "EXAM_HAS_FINDING",       # 检查提示某所见    EXAM_TEST → FINDING
    "FINDING_LOCALIZES_TO",   # 所见定位于        FINDING → BODY_SITE
    "PATTERN_IS",             # 影像/病理模式     IMAGING_FINDING → IMAGING_PATTERN
    "COMPARED_TO_PRIOR",      # 与前片比较        IMAGING_FINDING → IMAGING_FINDING
    "INDICATES",              # 提示诊断         LAB_RESULT/IMAGING_FINDING → DISEASE
    "NORMAL_FINDING",         # 检查/指标正常     EXAM_TEST/METRIC → RESULT/FINDING
    "ABNORMAL_FINDING",       # 检查/指标异常     EXAM_TEST/METRIC → RESULT/FINDING

    # 治疗关系
    "TREATED_WITH",           # 接受药物          PATIENT → DRUG
    "MEDICATION_FOR",         # 药物用于治疗      DRUG → DISEASE/SYMPTOM
    "UNDERWENT",              # 接受操作          PATIENT → PROCEDURE
    "PROCEDURE_FOR",          # 操作用于治疗/处理 PROCEDURE → DISEASE/SYMPTOM
    "DOSE_IS",                # 剂量用法          DRUG → DRUG_DOSE
    "HAS_DOSE",               # 药物剂量          DRUG → DRUG_DOSE
    "HAS_ROUTE",              # 给药途径          DRUG → DRUG_ROUTE
    "HAS_FREQUENCY",          # 给药频率          DRUG → DRUG_FREQUENCY/FREQUENCY
    "TREATMENT_FOR",          # 治疗目标疾病      DRUG/PROCEDURE → DISEASE
    "RESULTED_IN",            # 治疗效果          TREATMENT_PLAN → TREATMENT_RESPONSE
    "PARTICIPATED_IN",        # 参加临床试验      PATIENT → CLINICAL_TRIAL
    "HAS_ADHERENCE",          # 药物/治疗依从性   PATIENT/DRUG/TREATMENT_PLAN → MEDICATION_ADHERENCE
    "RECEIVED_VACCINATION",   # 接种疫苗         PATIENT → VACCINATION
    "CONSULTED_FOR",          # 会诊/MDT 针对问题 CONSULTATION → DISEASE/SYMPTOM

    # 暴露关系
    "EXPOSED_TO",             # 有暴露史          PATIENT → EXPOSURE
    "HAS_HABIT",              # 有生活习惯        PATIENT → HABIT
    "AMOUNT_IS",              # 暴露/习惯的量     EXPOSURE/HABIT → EXPOSURE_AMOUNT
    "DURATION_IS",            # 暴露/习惯的时长   EXPOSURE/HABIT → EXPOSURE_DURATION
    "RISK_FACTOR_FOR",        # 危险因素          EXPOSURE → DISEASE
]

# 每种 EdgeType 的语义说明（供 prompt 使用）
EDGE_TYPE_DESCRIPTIONS: dict[str, str] = {
    "HAS_SEX":               "患者的人口学性别",
    "HAS_AGE":               "患者的人口学年龄",
    "HAS_OCCUPATION":        "患者职业或长期工作",
    "HAS_BIRTHPLACE":        "患者出生地或籍贯",
    "HAS_GENERAL_STATUS":    "患者一般健康状态或病程中一般状态",
    "HAS_STATUS_ITEM":       "患者有某个一般状态观察项目",
    "STATUS_VALUE_IS":       "一般状态观察项目的取值",
    "HAS_SYMPTOM":           "患者有某症状，可来自主诉、现病史或病程",
    "SYMPTOM_ONSET":         "患者出现某症状，源节点通常是患者或时间，目标是 SYMPTOM",
    "HAS_SIGN":              "患者查体发现某体征，目标是 SIGN",
    "HAS_VITAL_SIGN":        "患者有某生命体征或床旁测量指标",
    "NEGATED_FINDING":       "患者否认某症状、体征、异常或疾病，边属性 negated=True",
    "SYMPTOM_WORSENS_WITH":  "某诱因/情境导致症状加重",
    "SYMPTOM_RELIEVES_WITH": "某因素使症状缓解",
    "WORSENS_AT_OR_FOR":     "症状在某时间点或最近某时长内加重",
    "WORSENS_WITH":          "某诱因、活动或情境导致症状加重",
    "RELIEVES_WITH":         "症状因某治疗、药物或因素缓解",
    "ACCOMPANIED_BY":        "症状A伴随症状B同时出现",
    "QUALIFIED_BY":          "实体被某修饰词描述（如胸闷气短→夜间为著）",
    "HAS_FUNCTIONAL_STATUS": "患者功能状态或活动耐量",
    "QUANTITY_OF":           "修饰词描述某实体的「量」",
    "COLOR_OF":              "修饰词描述某实体的「颜色」",
    "TEXTURE_OF":            "修饰词描述某实体的「性状/质地」",
    "TIMING_OF":             "修饰词描述症状发生的时间特征（如夜间、活动后）",
    "LOCALIZES_TO":          "症状/所见定位到某解剖部位",
    "MEASURED_VALUE_IS":     "生命体征、血气、肺功能或检验指标的具体数值",
    "SEVERITY_IS":           "症状/指标/功能损害的严重程度",
    "OCCURS_AT":             "事件发生的时间参照点",
    "LASTS_FOR":             "事件/状态持续的时长",
    "TEMPORAL_PRECEDES":     "事件A在时间上早于事件B",
    "TRIGGERED_BY":          "症状/发作被某诱因引起",
    "AFTER_EVENT":           "症状、疾病或事件发生在另一临床事件之后",
    "DIAGNOSED_WITH":        "患者被明确诊断为某疾病",
    "SUSPECTED_DIAGNOSIS":   "医生考虑/怀疑某诊断，尚未明确",
    "DIAGNOSIS_PENDING":     "疾病被列为待诊或待排",
    "DIAGNOSIS_ASSERTS":     "诊断陈述条目指向具体疾病",
    "DIFFERENTIAL_FOR":      "疾病A是疾病B的鉴别诊断对象",
    "SUPPORTED_BY":          "某检查/结果支持某诊断",
    "CONTRADICTS":           "某发现不支持某诊断",
    "DIAGNOSIS_CHANGED_TO":  "诊断从A更新为B",
    "COMORBIDITY":           "合并症，与主病并存",
    "HISTORY_OF":            "患者既往曾患某病或接受某操作",
    "NEGATED_HISTORY_OF":    "患者明确否认某既往病史、手术史、过敏史或传染病史",
    "CAUSED_BY":             "疾病/症状继发于另一疾病或暴露",
    "INFECTION_WITH":        "患者或疾病与病原体/感染事件相关",
    "UNDERWENT_EXAM":        "患者接受或完善某项检查",
    "HAS_RESULT":            "某检查项目对应的结果值",
    "HAS_FINDING":           "检查或部位存在某临床所见",
    "EXAM_HAS_FINDING":      "某项检查提示或显示某所见",
    "FINDING_LOCALIZES_TO":  "某所见定位于具体解剖部位",
    "PATTERN_IS":            "影像所见符合某特定模式",
    "COMPARED_TO_PRIOR":     "影像与既往相比的变化",
    "INDICATES":             "检查结果提示某诊断",
    "NORMAL_FINDING":        "检查、指标或所见被描述为正常/未见异常/阴性",
    "ABNORMAL_FINDING":      "检查、指标或所见被描述为异常/增高/减低/阳性",
    "TREATED_WITH":          "接受某药物治疗",
    "MEDICATION_FOR":        "药物用于治疗或控制某疾病/症状",
    "UNDERWENT":             "接受某操作/手术",
    "PROCEDURE_FOR":         "操作/手术针对某疾病或症状",
    "DOSE_IS":               "药物的剂量与用法",
    "HAS_DOSE":              "药物的单次剂量或剂量规格",
    "HAS_ROUTE":             "药物给药途径",
    "HAS_FREQUENCY":         "药物给药频率",
    "TREATMENT_FOR":         "治疗措施针对的目标疾病",
    "RESULTED_IN":           "治疗后的效果/转归",
    "PARTICIPATED_IN":       "患者参加临床试验或研究项目",
    "HAS_ADHERENCE":         "药物或治疗的规律性、持续性、依从性",
    "RECEIVED_VACCINATION":  "患者接受疫苗接种",
    "CONSULTED_FOR":         "会诊或 MDT 讨论针对某疾病/症状",
    "EXPOSED_TO":            "患者有某暴露史",
    "HAS_HABIT":             "患者有吸烟、饮酒、戒烟等生活习惯",
    "AMOUNT_IS":             "暴露或生活习惯的量",
    "DURATION_IS":           "暴露或生活习惯持续时间",
    "RISK_FACTOR_FOR":       "某暴露是某疾病的危险因素",
}


# 每种 EdgeType 的硬端点约束：source node_type 集合、target node_type 集合。
# C-4 验证器会使用该表直接删除方向或端点类型错误的边。
_PATIENT = frozenset({"PATIENT"})
_DEMOGRAPHIC = frozenset({"SEX", "AGE", "OCCUPATION", "BIRTHPLACE"})
_SYMPTOM = frozenset({"SYMPTOM"})
_SIGN = frozenset({"SIGN"})
_DISEASE = frozenset({"DISEASE", "DISEASE_SUBTYPE", "PATHOGEN_OR_INFECTION"})
_FINDING = frozenset({
    "SIGN",
    "IMAGING_FINDING",
    "CARDIAC_FINDING",
    "ECG_FINDING",
    "ULTRASOUND_FINDING",
    "PATHOLOGY_FINDING",
    "PFT_RESULT",
    "LAB_RESULT",
    "BLOOD_GAS_RESULT",
})
_RESULT = frozenset({"LAB_RESULT", "BLOOD_GAS_RESULT", "PFT_RESULT"})
_TEST_OR_METRIC = frozenset({
    "EXAM_TEST",
    "LAB_TEST",
    "BLOOD_GAS_METRIC",
    "PFT_METRIC",
    "VITAL_SIGN",
    "AUTOANTIBODY",
})
_CLINICAL_SUBJECT = frozenset({
    "PATIENT",
    "DISEASE",
    "DISEASE_SUBTYPE",
    "SYMPTOM",
    "SIGN",
    "IMAGING_FINDING",
    "CARDIAC_FINDING",
    "ECG_FINDING",
    "ULTRASOUND_FINDING",
    "PATHOLOGY_FINDING",
    "PFT_RESULT",
    "LAB_RESULT",
    "BLOOD_GAS_RESULT",
    "VITAL_SIGN",
    "FUNCTIONAL_STATUS",
    "GENERAL_STATUS_ITEM",
    "GENERAL_STATUS_VALUE",
})
_EVENT = frozenset({
    "CLINICAL_EVENT",
    "PROCEDURE",
    "OXYGEN_THERAPY",
    "CLINICAL_TRIAL",
    "CONSULTATION",
    "VACCINATION",
    "PATHOGEN_OR_INFECTION",
    "EXAM_TEST",
    "TREATMENT_PLAN",
})
_TIME_OR_DURATION = frozenset({"TIME_REF", "DURATION"})
_TREATMENT = frozenset({"DRUG", "TREATMENT_PLAN", "OXYGEN_THERAPY", "PROCEDURE"})
_THERAPY_OR_DRUG = frozenset({"DRUG", "TREATMENT_PLAN", "OXYGEN_THERAPY"})
_HABIT_OR_EXPOSURE = frozenset({"HABIT", "EXPOSURE"})
_FINDING_OR_RESULT = _FINDING | _RESULT | _TEST_OR_METRIC
_CLINICAL_EVENT_SOURCE = _CLINICAL_SUBJECT | _EVENT | _TREATMENT | _HABIT_OR_EXPOSURE

EDGE_ENDPOINT_CONSTRAINTS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    # 主体 / 人口学 / 一般状态
    "HAS_SEX": (_PATIENT, frozenset({"SEX"})),
    "HAS_AGE": (_PATIENT, frozenset({"AGE"})),
    "HAS_OCCUPATION": (_PATIENT, frozenset({"OCCUPATION"})),
    "HAS_BIRTHPLACE": (_PATIENT, frozenset({"BIRTHPLACE"})),
    "HAS_GENERAL_STATUS": (_PATIENT, frozenset({"GENERAL_HEALTH_STATUS"})),
    "HAS_STATUS_ITEM": (_PATIENT, frozenset({"GENERAL_STATUS_ITEM"})),
    "STATUS_VALUE_IS": (frozenset({"GENERAL_STATUS_ITEM"}), frozenset({"GENERAL_STATUS_VALUE"})),

    # 症状 / 体征关系
    "HAS_SYMPTOM": (_PATIENT | frozenset({"CLINICAL_EVENT", "TIME_REF"}), _SYMPTOM),
    "SYMPTOM_ONSET": (_PATIENT | frozenset({"CLINICAL_EVENT", "TIME_REF"}), _SYMPTOM),
    "HAS_SIGN": (_PATIENT | frozenset({"EXAM_TEST"}), _SIGN),
    "HAS_VITAL_SIGN": (_PATIENT, frozenset({"VITAL_SIGN"})),
    "HAS_FUNCTIONAL_STATUS": (_PATIENT, frozenset({"FUNCTIONAL_STATUS"})),
    "NEGATED_FINDING": (
        _PATIENT | _TEST_OR_METRIC,
        _SYMPTOM | _SIGN | _DISEASE | _FINDING | _RESULT | frozenset({"ALLERGY"}),
    ),
    "SYMPTOM_WORSENS_WITH": (
        _SYMPTOM | frozenset({"FUNCTIONAL_STATUS"}),
        frozenset({"TRIGGER", "QUALIFIER", "CLINICAL_EVENT", "FUNCTIONAL_STATUS"}),
    ),
    "SYMPTOM_RELIEVES_WITH": (
        _SYMPTOM,
        _THERAPY_OR_DRUG | frozenset({"QUALIFIER", "TREATMENT_RESPONSE"}),
    ),
    "WORSENS_AT_OR_FOR": (_SYMPTOM | frozenset({"DISEASE", "FUNCTIONAL_STATUS"}), _TIME_OR_DURATION),
    "WORSENS_WITH": (
        _SYMPTOM | frozenset({"DISEASE", "FUNCTIONAL_STATUS"}),
        frozenset({"TRIGGER", "QUALIFIER", "CLINICAL_EVENT", "FUNCTIONAL_STATUS"}),
    ),
    "RELIEVES_WITH": (_SYMPTOM | frozenset({"DISEASE"}), _THERAPY_OR_DRUG | frozenset({"QUALIFIER"})),
    "ACCOMPANIED_BY": (_SYMPTOM | _SIGN | _DISEASE, _SYMPTOM | _SIGN | _DISEASE),
    "QUALIFIED_BY": (
        _CLINICAL_SUBJECT | _TEST_OR_METRIC | frozenset({"DRUG", "TREATMENT_PLAN"}),
        frozenset({"QUALIFIER", "SEVERITY", "FREQUENCY"}),
    ),
    "QUANTITY_OF": (frozenset({"QUALIFIER"}), _SYMPTOM | _SIGN | _FINDING),
    "COLOR_OF": (frozenset({"QUALIFIER"}), _SYMPTOM | _SIGN | _FINDING),
    "TEXTURE_OF": (frozenset({"QUALIFIER"}), _SYMPTOM | _SIGN | _FINDING),
    "TIMING_OF": (frozenset({"QUALIFIER", "TIME_REF"}), _SYMPTOM | _SIGN | _EVENT),
    "LOCALIZES_TO": (_SYMPTOM | _SIGN | _FINDING | frozenset({"PROCEDURE"}), frozenset({"BODY_SITE"})),
    "MEASURED_VALUE_IS": (_TEST_OR_METRIC, _RESULT | frozenset({"LAB_RESULT", "BLOOD_GAS_RESULT"})),
    "SEVERITY_IS": (_CLINICAL_SUBJECT | _TEST_OR_METRIC | _RESULT, frozenset({"SEVERITY"})),

    # 时间关系
    "OCCURS_AT": (_CLINICAL_EVENT_SOURCE | _TEST_OR_METRIC, frozenset({"TIME_REF"})),
    "LASTS_FOR": (_CLINICAL_EVENT_SOURCE | _DISEASE, frozenset({"DURATION"})),
    "TEMPORAL_PRECEDES": (_CLINICAL_EVENT_SOURCE, _CLINICAL_EVENT_SOURCE),
    "TRIGGERED_BY": (_SYMPTOM | _DISEASE | _EVENT, frozenset({"TRIGGER", "PATHOGEN_OR_INFECTION", "VACCINATION", "EXPOSURE"})),
    "AFTER_EVENT": (_CLINICAL_EVENT_SOURCE, _EVENT | frozenset({"TIME_REF"})),

    # 诊断关系
    "DIAGNOSED_WITH": (_PATIENT | frozenset({"DIAGNOSIS_ASSERTION"}), _DISEASE),
    "SUSPECTED_DIAGNOSIS": (_PATIENT | frozenset({"DIAGNOSIS_ASSERTION", "IMAGING_FINDING"}), _DISEASE),
    "DIAGNOSIS_PENDING": (_PATIENT | frozenset({"DIAGNOSIS_ASSERTION"}), _DISEASE),
    "DIAGNOSIS_ASSERTS": (frozenset({"DIAGNOSIS_ASSERTION"}), _DISEASE),
    "DIFFERENTIAL_FOR": (_DISEASE, _DISEASE),
    "SUPPORTED_BY": (_DISEASE, _FINDING_OR_RESULT),
    "CONTRADICTS": (_DISEASE | _FINDING_OR_RESULT, _DISEASE | _FINDING_OR_RESULT),
    "DIAGNOSIS_CHANGED_TO": (_DISEASE, _DISEASE),

    # 合并症 / 病史关系
    "COMORBIDITY": (_PATIENT, _DISEASE),
    "HISTORY_OF": (_PATIENT, _DISEASE | frozenset({"PROCEDURE", "ALLERGY", "FAMILY_HISTORY_ITEM"})),
    "NEGATED_HISTORY_OF": (_PATIENT, _DISEASE | frozenset({"PROCEDURE", "ALLERGY", "FAMILY_HISTORY_ITEM"})),
    "CAUSED_BY": (_DISEASE | _SYMPTOM | _FINDING, _DISEASE | _HABIT_OR_EXPOSURE | frozenset({"PATHOGEN", "PATHOGEN_OR_INFECTION", "TRIGGER"})),
    "INFECTION_WITH": (_PATIENT | _DISEASE | _SYMPTOM, frozenset({"PATHOGEN", "PATHOGEN_OR_INFECTION"})),

    # 检查关系
    "UNDERWENT_EXAM": (_PATIENT, frozenset({"EXAM_TEST"})),
    "HAS_RESULT": (_TEST_OR_METRIC, _RESULT | frozenset({"LAB_RESULT", "BLOOD_GAS_RESULT"})),
    "HAS_FINDING": (frozenset({"EXAM_TEST", "BODY_SITE"}), _FINDING),
    "EXAM_HAS_FINDING": (frozenset({"EXAM_TEST"}), _FINDING),
    "FINDING_LOCALIZES_TO": (_FINDING, frozenset({"BODY_SITE"})),
    "PATTERN_IS": (_FINDING | frozenset({"PFT_RESULT"}), frozenset({"IMAGING_PATTERN"})),
    "COMPARED_TO_PRIOR": (_FINDING, _FINDING | frozenset({"TIME_REF"})),
    "INDICATES": (_FINDING_OR_RESULT, _DISEASE),
    "NORMAL_FINDING": (_TEST_OR_METRIC | _FINDING, _RESULT | _FINDING),
    "ABNORMAL_FINDING": (_TEST_OR_METRIC | _FINDING, _RESULT | _FINDING),

    # 治疗关系
    "TREATED_WITH": (_PATIENT | _DISEASE | _SYMPTOM, _THERAPY_OR_DRUG),
    "MEDICATION_FOR": (frozenset({"DRUG"}), _DISEASE | _SYMPTOM),
    "UNDERWENT": (_PATIENT, frozenset({"PROCEDURE"})),
    "PROCEDURE_FOR": (frozenset({"PROCEDURE", "OXYGEN_THERAPY", "CONSULTATION"}), _DISEASE | _SYMPTOM),
    "DOSE_IS": (frozenset({"DRUG"}), frozenset({"DRUG_DOSE"})),
    "HAS_DOSE": (frozenset({"DRUG"}), frozenset({"DRUG_DOSE"})),
    "HAS_ROUTE": (frozenset({"DRUG"}), frozenset({"DRUG_ROUTE"})),
    "HAS_FREQUENCY": (frozenset({"DRUG"}), frozenset({"DRUG_FREQUENCY", "FREQUENCY"})),
    "TREATMENT_FOR": (_TREATMENT | frozenset({"CONSULTATION"}), _DISEASE | _SYMPTOM),
    "RESULTED_IN": (_TREATMENT | frozenset({"CLINICAL_TRIAL"}), frozenset({"TREATMENT_RESPONSE", "GENERAL_HEALTH_STATUS", "SYMPTOM"})),
    "PARTICIPATED_IN": (_PATIENT, frozenset({"CLINICAL_TRIAL"})),
    "HAS_ADHERENCE": (_PATIENT | _THERAPY_OR_DRUG, frozenset({"MEDICATION_ADHERENCE"})),
    "RECEIVED_VACCINATION": (_PATIENT, frozenset({"VACCINATION"})),
    "CONSULTED_FOR": (frozenset({"CONSULTATION"}), _DISEASE | _SYMPTOM),

    # 暴露关系
    "EXPOSED_TO": (_PATIENT, frozenset({"EXPOSURE"})),
    "HAS_HABIT": (_PATIENT, frozenset({"HABIT"})),
    "AMOUNT_IS": (_HABIT_OR_EXPOSURE, frozenset({"EXPOSURE_AMOUNT"})),
    "DURATION_IS": (_HABIT_OR_EXPOSURE, frozenset({"EXPOSURE_DURATION", "DURATION"})),
    "RISK_FACTOR_FOR": (_HABIT_OR_EXPOSURE, _DISEASE),
}


# ---------------------------------------------------------------------------
# Certainty
# ---------------------------------------------------------------------------

Certainty = Literal["confirmed", "suspected", "excluded", "uncertain"]


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class GraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(..., description="临时 ID，格式 n{i}，从 1 开始")
    node_type: NodeType
    source_text: str = Field(
        ...,
        description="原文中对应该实体的精确连续子串，不得改写",
    )
    negated: bool = Field(
        False,
        description="该实体在原文语境中是否被否定（由 clinical_context_preprocessor 预标注）",
    )
    certainty: Certainty = Field(
        "confirmed",
        description="该实体在临床语境中的确定性",
    )
    implicit: bool = Field(
        False,
        description="是否为代码注入的隐式节点；隐式 PATIENT 可不对应原文子串",
    )


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_id: str = Field(..., description="临时 ID，格式 e{i}，从 1 开始")
    edge_type: EdgeType
    source_node_id: str = Field(..., description="出发节点的 node_id")
    target_node_id: str = Field(..., description="目标节点的 node_id")
    negated: bool = Field(False, description="该关系是否在否定语境下成立")
    certainty: Certainty = "confirmed"
    context: str | None = Field(
        None,
        description="修饰该关系的短语（如「无明显诱因」、「活动后明显」），来自原文",
    )
    source_text: str = Field(
        ...,
        description="支持推断该关系的原文片段，可大致定位即可，不要求精确子串",
    )
    reasoning: str = Field(
        ...,
        description=(
            "LLM 解释：source_text 中的哪个表达支持选择 edge_type。"
            "格式示例：「source_text 中的「伴」表明两症状同时出现，符合 ACCOMPANIED_BY」"
        ),
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="LLM 自评对该边正确性的置信度，0~1",
    )


# ---------------------------------------------------------------------------
# Context frame
# ---------------------------------------------------------------------------

class GraphFrame(BaseModel):
    """图中的上下文框，如「病程中」包住一组状态断言。"""
    model_config = ConfigDict(extra="forbid")

    frame_id: str
    frame_type: Literal["temporal_context", "clinical_context"] = "clinical_context"
    source_text: str
    node_ids: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# EdgeValidationResult（C-4 验证器输出）
# ---------------------------------------------------------------------------

class EdgeValidationResult(BaseModel):
    """C-4 阶段 LLM 对单条边 reasoning 的审查结果。"""
    model_config = ConfigDict(extra="forbid")

    edge_id: str
    verdict: Literal["valid", "low_confidence", "invalid"]
    verdict_reason: str = Field(
        ...,
        description="一句话说明判定结果的依据，≤ 60 字",
    )


# ---------------------------------------------------------------------------
# ChunkGraph
# ---------------------------------------------------------------------------

class ChunkGraph(BaseModel):
    """单个 SemanticChunk 产出的局部子图。"""

    chunk_id: str
    case_id: str
    section_id: str
    section_index: int
    chunk_index: int
    chunk_type: str
    chunk_text: str = Field(..., description="对应 SemanticChunk.text，供验证器使用")
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    frames: list[GraphFrame] = Field(default_factory=list)

    # 验证后填充
    validation_warnings: list[str] = Field(default_factory=list)
    low_confidence_edge_ids: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return self.model_dump()


# ---------------------------------------------------------------------------
# LLM 内部输出 Schema（C-3 时使用，不含 chunk_id 等元信息）
# ---------------------------------------------------------------------------

class _GraphBuilderOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Chunk Analysis（C-3a 中间表示：纯自然语言分析，无 schema 约束）
# ---------------------------------------------------------------------------

class _AnalyzedEntity(BaseModel):
    """C-3a 输出的实体：纯文本描述，尚未映射到 NodeType。"""
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., description="原文中的精确连续子串")
    description: str = Field(
        ...,
        description="该实体在当前 chunk 上下文中的临床含义（一句话），用于后续 NodeType 选择",
    )
    negated: bool = False
    certainty: Certainty = "confirmed"


class _AnalyzedRelation(BaseModel):
    """C-3a 输出的关系：用自然语言描述主谓宾，尚未映射到 EdgeType。"""
    model_config = ConfigDict(extra="forbid")

    subject_text: str = Field(..., description="关系主语对应的实体 text（必须出现在 entities 列表）")
    object_text: str = Field(..., description="关系宾语对应的实体 text（必须出现在 entities 列表）")
    relation: str = Field(
        ...,
        description=(
            "用一句中文自然语言准确描述主语和宾语的临床关系，例如：「咳嗽咳痰持续了 8 年」、"
            "「咳嗽咳痰与胸闷气短同时存在（伴随）」、「症状最近 2 个月加重」。"
            "禁止使用 schema 名词如 SYMPTOM_ONSET、LASTS_FOR，必须是医生口语化表达。"
        ),
    )
    evidence_text: str = Field(..., description="支撑该关系的原文片段，尽量精确")
    negated: bool = False
    certainty: Certainty = "confirmed"


class _ChunkAnalysis(BaseModel):
    """C-3a：对一个 SemanticChunk 的纯语义分析结果，作为 C-3b 的输入。"""
    model_config = ConfigDict(extra="forbid")

    overview: str = Field(
        ...,
        description="一两句话用自然语言准确复述该 chunk 想表达的全部临床信息。",
    )
    entities: list[_AnalyzedEntity] = Field(default_factory=list)
    relations: list[_AnalyzedRelation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# SectionGraph（C-D 合并后）
# ---------------------------------------------------------------------------

class SectionGraphNode(BaseModel):
    """合并后的全局节点，node_id 替换为全局 ID。"""
    model_config = ConfigDict(extra="forbid")

    node_id: str              # 格式：{case_id}_sec{si:03d}_nd{ni:04d}
    node_type: NodeType
    source_text: str
    negated: bool = False
    certainty: Certainty = "confirmed"
    implicit: bool = False
    chunk_ids: list[str] = Field(default_factory=list, description="来源 chunk 列表")
    merged_from: list[str] = Field(
        default_factory=list,
        description="合并时被归并的临时 node_id 列表",
    )


class SectionGraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_id: str              # 格式：{case_id}_sec{si:03d}_ed{ei:04d}
    edge_type: EdgeType
    source_node_id: str
    target_node_id: str
    negated: bool = False
    certainty: Certainty = "confirmed"
    context: str | None = None
    source_text: str
    reasoning: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    chunk_ids: list[str] = Field(default_factory=list)
    low_confidence: bool = False


class SectionGraph(BaseModel):
    """一个 DiscourseSection 合并后的完整图。"""

    graph_id: str             # 格式：{case_id}_sec{si:03d}_graph
    case_id: str
    section_id: str
    section_index: int
    section_name: str
    nodes: list[SectionGraphNode] = Field(default_factory=list)
    edges: list[SectionGraphEdge] = Field(default_factory=list)
    frames: list[GraphFrame] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return self.model_dump()
