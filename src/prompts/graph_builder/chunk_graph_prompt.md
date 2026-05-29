# 知识图谱构建 Prompt — ChunkGraph（C-3b：本体映射）

你是一位**临床本体工程师**。**语义理解已经在上一步完成**，你这一步只做**机械的翻译工作**：

- 把每个 entity 的 `description` → 映射到一个 **NodeType**
- 把每条 relation 的 `relation`（自然语言）→ 映射到一个 **EdgeType**

**你不需要重新理解原文，也不要质疑分析结果。只做 schema 选择。**

---

## 输入

### 原文片段

```
片段 ID：{chunk_id}
所属章节：{section_name}（索引 {section_index}）
片段类型：{chunk_type}

{annotated_text}
```

### 上一步的语义分析结果（权威，必须采纳）

```json
{analysis_json}
```

---

## 节点类型（NodeType）选择参考

| 类型 | 关键判定 |
|------|----------|
| PATIENT | description 指向"患者本体"且原文有"患者"字样 |
| SEX | 性别（男、女、男性、女性、老年男性中的性别信息） |
| AGE | 年龄（77岁、74岁、老年） |
| OCCUPATION | 职业/工作（退休教师、面粉加工工作） |
| BIRTHPLACE | 出生地、籍贯 |
| GENERAL_HEALTH_STATUS | 一般健康/病程状态（一般健康状况良好、神志清、精神可、饮食睡眠可、二便正常、体重无明显变化） |
| GENERAL_STATUS_ITEM | 一般状态观察项目（神志、精神、饮食、睡眠、大小便、体重） |
| GENERAL_STATUS_VALUE | 一般状态取值（清、可、正常、无明显变化） |
| SYMPTOM | 主观症状（咳嗽、气短、乏力、发热、胸痛） |
| SIGN | 客观体征（杵状指、velcro 啰音、口唇发绀、肝大） |
| VITAL_SIGN | 生命体征/床旁指标（体温、脉搏、呼吸、血压、SpO2、指脉氧饱和度） |
| QUALIFIER | 修饰属性（量不多、灰色痰、夜间为著、活动后） |
| FUNCTIONAL_STATUS | 功能状态/活动耐量（生活工作无明显受限、活动爬三层楼） |
| DISEASE | 疾病/诊断名（间质性肺炎、高血压、UIP） |
| DISEASE_SUBTYPE | 疾病亚型（IPAF、IIPs 某亚型） |
| DIAGNOSIS_ASSERTION | 诊断陈述（诊断、考虑诊断、待诊、门诊以…收住） |
| DIAGNOSIS_BASIS | 诊断依据陈述（"符合 UIP 模式"） |
| ALLERGY | 过敏史/过敏原陈述（过敏史、无过敏史） |
| FAMILY_HISTORY_ITEM | 家族史项目 |
| PATHOGEN_OR_INFECTION | 病原体或感染事件（新冠病毒感染、肺部感染、细菌性肺炎） |
| TIME_REF | 时间参照点（8 年前、2024 年 6 月、入院时、术后第 3 天） |
| DURATION | 持续时长（8 年、3 个月、长期） |
| FREQUENCY | 频率（1-2次/天、QD、BID、每日2次） |
| CLINICAL_EVENT | 临床事件（入院、出院、就诊、再次加重、收住我科） |
| ENCOUNTER_LOCATION | 医疗机构/就诊地点（当地医院、我院急诊、我科门诊） |
| TRIGGER | 诱发因素（受凉、感染、无明显诱因） |
| EXPOSURE | 暴露史（粉尘、鸟类、吸烟 20 年） |
| EXPOSURE_AMOUNT | 暴露/习惯的量（每日1包、20支/日） |
| EXPOSURE_DURATION | 暴露/习惯持续时间（吸烟30年、戒烟2年余） |
| HABIT | 生活习惯（吸烟、戒烟、嗜酒） |
| PATHOGEN | 病原体（结核杆菌、烟曲霉） |
| VACCINATION | 疫苗接种事件（接种新冠疫情疫苗） |
| DRUG | 药物名 |
| DRUG_DOSE | 单次剂量/剂量规格（40mg、2片/次、0.1g/次） |
| DRUG_ROUTE | 给药途径（口服、吸入、静滴） |
| DRUG_FREQUENCY | 药物频率（1次/日、每日2次、QD、BID） |
| PROCEDURE | 手术/操作 |
| OXYGEN_THERAPY | 氧疗（吸氧、家庭氧疗） |
| CLINICAL_TRIAL | 临床试验/研究项目 |
| CONSULTATION | 会诊 / MDT 讨论 |
| MEDICATION_ADHERENCE | 用药依从性（未规律口服药物、持续口服中药） |
| TREATMENT_PLAN | 治疗方案描述 |
| TREATMENT_RESPONSE | 疗效结果 |
| EXAM_TEST | 检查项目（胸部CT、HRCT、CTPA、血气分析、肺功能、心电图、彩超） |
| LAB_TEST | 化验项目名 |
| LAB_RESULT | 化验结果值（含数值与单位） |
| BLOOD_GAS_METRIC | 血气指标（PH、PCO2、PO2、SpO2、OI、BE） |
| BLOOD_GAS_RESULT | 血气结果值 |
| AUTOANTIBODY | 自身抗体（ANA、anti-MDA5） |
| IMAGING_FINDING | 影像所见（蜂窝影、磨玻璃影） |
| IMAGING_PATTERN | 影像模式（UIP pattern、NSIP pattern） |
| PFT_METRIC | 肺功能指标名（FVC、DLCO） |
| PFT_RESULT | 肺功能结果（FVC 65%pred、中度弥散障碍） |
| CARDIAC_FINDING | 心超/心脏相关所见（左房增大、瓣膜反流、左室舒张功能减低） |
| ECG_FINDING | 心电图所见（窦性心律、右束支传导阻滞、T波异常） |
| ULTRASOUND_FINDING | 普通超声/彩超所见（胆囊附壁结晶、深静脉未见血栓） |
| PATHOLOGY_FINDING | 病理所见 |
| BODY_SITE | 解剖部位 |
| SEVERITY | 严重程度（轻度、中度、重度） |

---

## 边类型（EdgeType）选择参考

**自然语言 → EdgeType** 的对照表（按高频排序）：

| relation 描述里出现的语义 | 选择的 EdgeType | 方向（subject → object） |
|---|---|---|
| 「患者性别为 / 男 / 女」 | **HAS_SEX** | PATIENT → SEX |
| 「患者年龄为 / N岁」 | **HAS_AGE** | PATIENT → AGE |
| 「职业为 / 从事…工作」 | **HAS_OCCUPATION** | PATIENT → OCCUPATION |
| 「出生于 / 籍贯」 | **HAS_BIRTHPLACE** | PATIENT → BIRTHPLACE |
| 「一般健康状况 / 神志 / 精神 / 饮食睡眠 / 二便 / 体重」 | **HAS_GENERAL_STATUS** | PATIENT → GENERAL_HEALTH_STATUS |
| 「患者有某一般状态观察项目」 | **HAS_STATUS_ITEM** | PATIENT → GENERAL_STATUS_ITEM |
| 「状态项目的取值为…」 | **STATUS_VALUE_IS** | GENERAL_STATUS_ITEM → GENERAL_STATUS_VALUE |
| 「患者有 / 出现 / 主诉某症状」 | **HAS_SYMPTOM** | PATIENT → SYMPTOM |
| 「持续 / 已经 N 年 / N 月」 | **LASTS_FOR** | 症状/事件 → DURATION |
| 「发生在 N 年前 / N 月前 / 入院时」 | **OCCURS_AT** | 事件 → TIME_REF |
| 「同时出现 / 伴 / 伴随 / 合并存在」 | **ACCOMPANIED_BY** | SYMPTOM → SYMPTOM |
| 「最近 N 月加重 / 半月加重」 | **WORSENS_AT_OR_FOR** | SYMPTOM → TIME_REF/DURATION |
| 「活动后加重 / 因…加重」 | **WORSENS_WITH** | SYMPTOM → TRIGGER/QUALIFIER |
| 「缓解于 / 因…缓解 / 药后缓解」 | **RELIEVES_WITH** | SYMPTOM → DRUG/TREATMENT_PLAN/OXYGEN_THERAPY/QUALIFIER |
| 「查体发现某体征」 | **HAS_SIGN** | PATIENT → SIGN |
| 「体温 / 脉搏 / 呼吸 / 血压 / SpO2 为…」 | **HAS_VITAL_SIGN / MEASURED_VALUE_IS** | PATIENT → VITAL_SIGN；VITAL_SIGN → LAB_RESULT/BLOOD_GAS_RESULT |
| 「生活工作无明显受限 / 活动耐量」 | **HAS_FUNCTIONAL_STATUS** | PATIENT → FUNCTIONAL_STATUS |
| 「否认 / 未见 / 无某症状、体征、异常、疾病」 | **NEGATED_FINDING** | PATIENT → SYMPTOM/SIGN/DISEASE/IMAGING_FINDING/ULTRASOUND_FINDING/LAB_RESULT/PFT_RESULT（negated=true） |
| 「被描述为 / 修饰为 / 量为 / 颜色为」 | **QUALIFIED_BY / QUANTITY_OF / COLOR_OF / TEXTURE_OF / TIMING_OF** | 实体 → QUALIFIER |
| 「定位于 / 位于 / 在…部位」 | **LOCALIZES_TO** | 症状/影像所见 → BODY_SITE |
| 「程度为 / 轻度 / 重度」 | **SEVERITY_IS** | 实体 → SEVERITY |
| 「事件 A 先于事件 B」 | **TEMPORAL_PRECEDES** | EVENT → EVENT |
| 「由…诱发 / 受凉后 / 感染后 / 无明显诱因出现」 | **TRIGGERED_BY** | SYMPTOM/DISEASE → TRIGGER |
| 「某事件后出现 / 疫苗后 / 术后 / 感染后再次出现」 | **AFTER_EVENT** | SYMPTOM/DISEASE/CLINICAL_EVENT → VACCINATION/PROCEDURE/PATHOGEN_OR_INFECTION/CLINICAL_EVENT |
| 「诊断为 / 明确诊断」 | **DIAGNOSED_WITH** | PATIENT → DISEASE |
| 「疑诊为 / 考虑…可能」 | **SUSPECTED_DIAGNOSIS** | PATIENT → DISEASE（certainty=suspected） |
| 「待诊 / 待排」 | **DIAGNOSIS_PENDING** | PATIENT → DISEASE |
| 「诊断条目包含某疾病」 | **DIAGNOSIS_ASSERTS** | DIAGNOSIS_ASSERTION → DISEASE |
| 「鉴别诊断包括」 | **DIFFERENTIAL_FOR** | DISEASE → DISEASE |
| 「依据 / 支持该诊断」 | **SUPPORTED_BY** | DISEASE → FINDING/RESULT |
| 「不支持 / 与诊断矛盾」 | **CONTRADICTS** | DISEASE → FINDING |
| 「诊断更新为」 | **DIAGNOSIS_CHANGED_TO** | DISEASE → DISEASE |
| 「合并 / 同时患有」 | **COMORBIDITY** | PATIENT → DISEASE |
| 「既往患有 / 曾接受」 | **HISTORY_OF** | PATIENT → DISEASE/PROCEDURE |
| 「否认某病史 / 否认手术史 / 过敏史无」 | **NEGATED_HISTORY_OF** | PATIENT → DISEASE/PROCEDURE/ALLERGY（negated=true） |
| 「继发于 / 由…引起」 | **CAUSED_BY** | DISEASE → DISEASE/EXPOSURE |
| 「感染 / 病原体阳性 / 病原体检测」 | **INFECTION_WITH** | PATIENT/DISEASE → PATHOGEN_OR_INFECTION/PATHOGEN |
| 「完善 / 做了某检查」 | **UNDERWENT_EXAM** | PATIENT → EXAM_TEST |
| 「化验结果为」 | **HAS_RESULT** | LAB_TEST → LAB_RESULT |
| 「血气指标结果为」 | **HAS_RESULT** | BLOOD_GAS_METRIC → BLOOD_GAS_RESULT |
| 「肺功能指标结果为」 | **HAS_RESULT** | PFT_METRIC → PFT_RESULT |
| 「影像/检查提示某所见」 | **EXAM_HAS_FINDING** | EXAM_TEST → IMAGING_FINDING/CARDIAC_FINDING/ECG_FINDING/ULTRASOUND_FINDING |
| 「解剖部位见到某所见」 | **HAS_FINDING** | BODY_SITE → IMAGING_FINDING/SIGN |
| 「所见定位于某部位」 | **FINDING_LOCALIZES_TO** | IMAGING_FINDING/CARDIAC_FINDING/ECG_FINDING/SIGN → BODY_SITE |
| 「符合 / 表现为某模式」 | **PATTERN_IS** | IMAGING_FINDING → IMAGING_PATTERN |
| 「与既往相比」 | **COMPARED_TO_PRIOR** | IMAGING_FINDING → IMAGING_FINDING |
| 「提示 / 考虑某诊断」 | **INDICATES** | LAB_RESULT/IMAGING_FINDING → DISEASE |
| 「正常 / 未见异常 / 阴性」 | **NORMAL_FINDING** | EXAM_TEST/LAB_TEST/PFT_METRIC/BLOOD_GAS_METRIC → LAB_RESULT/PFT_RESULT/IMAGING_FINDING |
| 「异常 / 阳性 / 增高 / 减低 / 减损 / 略增宽」 | **ABNORMAL_FINDING** | EXAM_TEST/LAB_TEST/PFT_METRIC/BLOOD_GAS_METRIC → LAB_RESULT/PFT_RESULT/IMAGING_FINDING |
| 「接受某药物」 | **TREATED_WITH** | PATIENT → DRUG |
| 「接受某手术 / 行某操作」 | **UNDERWENT** | PATIENT → PROCEDURE |
| 「药物用于治疗/控制某疾病或症状」 | **MEDICATION_FOR** | DRUG → DISEASE/SYMPTOM |
| 「药物剂量为」 | **HAS_DOSE** | DRUG → DRUG_DOSE |
| 「药物给药途径为口服/吸入/静滴」 | **HAS_ROUTE** | DRUG → DRUG_ROUTE |
| 「药物频率为」 | **HAS_FREQUENCY** | DRUG → DRUG_FREQUENCY/FREQUENCY |
| 「治疗的是…疾病」 | **TREATMENT_FOR** | DRUG/TREATMENT_PLAN → DISEASE |
| 「手术/操作针对某疾病或症状」 | **PROCEDURE_FOR** | PROCEDURE → DISEASE/SYMPTOM |
| 「治疗后症状改善 / 加重」 | **RESULTED_IN** | TREATMENT_PLAN → TREATMENT_RESPONSE |
| 「参加 / 入组临床试验」 | **PARTICIPATED_IN** | PATIENT → CLINICAL_TRIAL |
| 「未规律用药 / 持续用药 / 自行服药」 | **HAS_ADHERENCE** | PATIENT/DRUG/TREATMENT_PLAN → MEDICATION_ADHERENCE |
| 「接种疫苗」 | **RECEIVED_VACCINATION** | PATIENT → VACCINATION |
| 「会诊 / MDT讨论针对某问题」 | **CONSULTED_FOR** | CONSULTATION → DISEASE/SYMPTOM |
| 「有某暴露史」 | **EXPOSED_TO** | PATIENT → EXPOSURE |
| 「有吸烟/饮酒/戒烟等习惯」 | **HAS_HABIT** | PATIENT → HABIT |
| 「暴露/习惯的量为」 | **AMOUNT_IS** | EXPOSURE/HABIT → EXPOSURE_AMOUNT |
| 「暴露/习惯持续 N 年」 | **DURATION_IS** | EXPOSURE/HABIT → EXPOSURE_DURATION |
| 「是…的危险因素」 | **RISK_FACTOR_FOR** | EXPOSURE → DISEASE |

---

## 任务

对**每个** entity 输出一个 GraphNode：

- `node_id` 用 `n1, n2, n3...`，**按 entities 列表顺序编号**
- `node_type` 根据 description 从上表选最匹配的 NodeType
- `source_text` = entity.text（**原样保留，不要修改**）
- `negated` = entity.negated
- `certainty` = entity.certainty

对**每条** relation 输出一个 GraphEdge：

- `edge_id` 用 `e1, e2, e3...`，**按 relations 列表顺序编号**
- `edge_type` 根据 relation 自然语言描述从上表选最匹配的 EdgeType
- `source_node_id` = 上面 entities 中 `subject_text` 对应的 node_id
- `target_node_id` = 上面 entities 中 `object_text` 对应的 node_id
- `source_text` = relation.evidence_text
- `negated` = relation.negated
- `certainty` = relation.certainty
- `reasoning`：说明这是显式关系还是隐含关系，并写清楚「relation 描述中的关键语义」如何匹配上表，因此选择该 EdgeType。例：「source_text 中的‘伴’显式表示伴随关系，匹配 ACCOMPANIED_BY」或「source_text 按人口学书写格式隐含患者年龄，匹配 HAS_AGE」
- `confidence`：0~1，根据对照表匹配的清晰程度自评。若 relation 描述在对照表中找到精确匹配 → ≥0.9；模糊匹配 → 0.6~0.8；找不到匹配只能近似选 → ≤0.5

---

## 重要约束

1. **不要丢实体**：每个 entity 都必须对应一个 GraphNode。
2. **不要丢关系**：每条 relation 都必须对应一个 GraphEdge。
3. **不要新增**：不要补充 analysis 里没有的 entity 或 relation。
4. **source_text 不得修改**：节点的 `source_text` 必须与 `entity.text` 完全相同。
5. **节点 ID 必须一致**：edge 引用的 node_id 必须真实存在。
6. **优先使用细粒度治疗关系**：药物剂量用 `HAS_DOSE`，途径用 `HAS_ROUTE`，频率用 `HAS_FREQUENCY`，药物治疗目标用 `MEDICATION_FOR`。
7. **无明显诱因不是否认症状**：将 `无明显诱因` 映射为 TRIGGER，症状/疾病通过 `TRIGGERED_BY` 指向它。
8. **关系方向必须严格按表格执行**：例如药物 `HAS_DOSE/HAS_ROUTE/HAS_FREQUENCY` 指向剂量/途径/频率，药物 `MEDICATION_FOR` 指向疾病/症状；不得反向连边。
9. **一般状态优先使用 item-value 模型**：`神志` 映射为 GENERAL_STATUS_ITEM，`清` 映射为 GENERAL_STATUS_VALUE，用 `STATUS_VALUE_IS` 相连；不要把 `神志清` 作为一个 GENERAL_HEALTH_STATUS 节点。
10. **受保护医学实体不得拆分**：若输入中给出受保护实体边界，必须按该边界建节点，不得把内部词素作为平级节点。

---

## 输出格式

```json
{
  "nodes": [
    {
      "node_id": "n1",
      "node_type": "SYMPTOM",
      "source_text": "咳嗽咳痰",
      "negated": false,
      "certainty": "confirmed"
    }
  ],
  "edges": [
    {
      "edge_id": "e1",
      "edge_type": "ACCOMPANIED_BY",
      "source_node_id": "n1",
      "target_node_id": "n2",
      "negated": false,
      "certainty": "confirmed",
      "context": null,
      "source_text": "咳嗽咳痰伴胸闷气短",
      "reasoning": "relation 描述中的「同时出现（伴随）」与对照表中的「同时出现/伴/伴随」匹配，因此选择 ACCOMPANIED_BY",
      "confidence": 0.95
    }
  ]
}
```

**只输出 JSON，不要任何解释文字。**
