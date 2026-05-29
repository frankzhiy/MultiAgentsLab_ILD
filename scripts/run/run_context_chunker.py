"""Runner for the all-LLM discourse sectionizer + graph builder pipeline.

输出：
  outputs/runs/{run_id}_{case_id}/
    ├── {case_id}_sections.json              # Step A+B 全 phase 数据
    ├── {case_id}_sections.html             # Step A+B+C 可视化（含图谱摘要）
    └── {case_id}_sec###_graph.json         # 每个 section 的 SectionGraph（Step C）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from html import escape
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.context_chunker.discourse_pipeline import (  # noqa: E402
    DiscourseSectionedCase,
    section_raw_text_async,
)
from src.agents.graph_builder.chunk_filter import get_graphable_chunks  # noqa: E402
from src.agents.graph_builder.chunk_graph_builder import build_chunk_graphs_async  # noqa: E402
from src.agents.graph_builder.chunk_graph_validator import validate_chunk_graphs_async  # noqa: E402
from src.agents.graph_builder.clinical_context_preprocessor import annotate_chunks  # noqa: E402
from src.agents.graph_builder.section_graph_merger import merge_section_graph_async  # noqa: E402
from src.schemas.context_chunker.discourse_taxonomy import DISCOURSE_TAXONOMY  # noqa: E402
from src.schemas.graph.clinical_graph import EDGE_TYPE_DESCRIPTIONS, SectionGraph  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logger(log_dir: Path, run_id: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_id}.log"

    logger = logging.getLogger("context_chunker_runner")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    cli_handler = logging.StreamHandler(sys.stdout)
    cli_handler.setLevel(logging.INFO)
    cli_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )

    logger.addHandler(cli_handler)
    logger.addHandler(file_handler)
    return logger


# ---------------------------------------------------------------------------
# Interactive file selection
# ---------------------------------------------------------------------------

def _select_files_interactively(txt_files: list[Path]) -> list[Path]:
    print("\n可用的原始病例文件：")
    for i, f in enumerate(txt_files, 1):
        print(f"  [{i}] {f.name}")
    print(f"  [0] 全部处理（共 {len(txt_files)} 个文件）\n")

    while True:
        try:
            raw = input("请输入序号（多个用英文逗号分隔，0 表示全部）：").strip()
        except (EOFError, KeyboardInterrupt) as e:
            raise SystemExit("\n已中止。") from e
        if not raw:
            continue
        if raw == "0":
            return list(txt_files)
        try:
            indices = [int(x.strip()) for x in raw.split(",")]
        except ValueError:
            print("  ✗ 无效输入。\n")
            continue
        if not all(1 <= idx <= len(txt_files) for idx in indices):
            print(f"  ✗ 序号超出范围 1–{len(txt_files)}\n")
            continue
        seen: set[int] = set()
        selected: list[Path] = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                selected.append(txt_files[idx - 1])
        return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run all-LLM discourse sectionizer.")
    parser.add_argument("--input", default="data/raw_cases")
    parser.add_argument("--output", default="outputs/runs")
    parser.add_argument("--all", dest="process_all", action="store_true")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "logs"
    logger = _setup_logger(log_dir, f"run_{run_id}")

    input_dir = _resolve_project_path(args.input)
    output_root = _resolve_project_path(args.output)

    logger.info("=" * 60)
    logger.info(f"context_chunker (LLM) 运行开始  run_id={run_id}")
    logger.info(f"输入目录 : {input_dir}")
    logger.info(f"输出根目录: {output_root}")
    logger.info(f"日志文件 : {log_dir / f'run_{run_id}.log'}")
    logger.info("=" * 60)

    txt_files = sorted(f for f in input_dir.glob("*.txt") if not f.name.startswith("."))
    if not txt_files:
        logger.error(f"在输入目录中未找到 .txt 文件：{input_dir}")
        raise SystemExit(1)

    logger.info(f"发现 {len(txt_files)} 个病例文件：{[f.name for f in txt_files]}")

    selected = txt_files if args.process_all else _select_files_interactively(txt_files)
    logger.info(f"已选择 {len(selected)} 个：{[f.name for f in selected]}")

    asyncio.run(_run_all(selected, output_root, run_id, logger))


# ---------------------------------------------------------------------------
# Step C helper
# ---------------------------------------------------------------------------

async def _run_step_c_async(
    case: DiscourseSectionedCase,
    out_dir: Path,
    logger: logging.Logger,
) -> list[SectionGraph]:
    """Step C: filter → annotate → build → validate → merge，所有 section 并行执行。

    全局共享一个 Semaphore，限制总 LLM 并发数，避免撞 rate limit。
    """
    if not case.chunks:
        logger.warning("  Step C: 无 chunk，跳过")
        return []

    from collections import defaultdict

    # 读取 max_concurrency 配置，创建全局 Semaphore
    import yaml as _yaml
    from src.agents.context_chunker.config_loader import resolve_project_path as _rp
    _cfg_path = _rp("configs/agents/graph_builder/graph_builder.yaml")
    _cfg = _yaml.safe_load(_cfg_path.read_text(encoding="utf-8")) if _cfg_path.exists() else {}
    _max_conc: int = _cfg.get("chunk_graph_builder", {}).get("max_concurrency", 4)
    semaphore = asyncio.Semaphore(_max_conc)

    sections_by_index = {sec.index: sec for sec in case.sections}
    chunks_by_section: dict[int, list] = defaultdict(list)
    for chunk in sorted(case.chunks, key=lambda c: c.section_index):
        chunks_by_section[chunk.section_index].append(chunk)

    async def _process_section(sec_idx: int, sec_chunks: list) -> SectionGraph | None:
        sec = sections_by_index.get(sec_idx)
        sec_name = sec.section_name if sec else f"section_{sec_idx}"
        logger.info(f"    [{sec_idx}] {sec_name}: {len(sec_chunks)} chunks")

        graphable = get_graphable_chunks(sec_chunks)
        if not graphable:
            logger.info(f"      [{sec_idx}] 全部 chunk 被过滤，跳过")
            return None

        annotated = annotate_chunks(graphable)
        chunk_debug_dir = out_dir / "chunk_debug" / f"sec{sec_idx:03d}"
        chunk_graphs = await build_chunk_graphs_async(
            annotated,
            semaphore=semaphore,
            debug_output_dir=chunk_debug_dir,
        )
        chunk_graphs = await validate_chunk_graphs_async(chunk_graphs, semaphore=semaphore)

        section_graph, _ = await merge_section_graph_async(
            chunk_graphs,
            section_name=sec_name,
            output_dir=out_dir,
            semaphore=semaphore,
        )
        logger.info(
            f"      [{sec_idx}] 节点 {len(section_graph.nodes)}, 边 {len(section_graph.edges)}"
        )
        return section_graph

    tasks = [
        _process_section(sec_idx, sec_chunks)
        for sec_idx, sec_chunks in sorted(chunks_by_section.items())
    ]
    results = await asyncio.gather(*tasks)
    return [g for g in results if g is not None]


async def _run_all(
    selected: list[Path],
    output_root: Path,
    run_id: str,
    logger: logging.Logger,
) -> None:
    run_wall_start = time.perf_counter()
    for file_idx, txt_path in enumerate(selected, 1):
        logger.info("-" * 60)
        logger.info(f"[{file_idx}/{len(selected)}] 开始处理：{txt_path.name}")
        t0 = time.perf_counter()

        raw_text = txt_path.read_text(encoding="utf-8")
        logger.info(f"  读取完成，文本长度 {len(raw_text)} 字符")

        case = await section_raw_text_async(raw_text, logger=logger)

        case_out_dir = output_root / f"{run_id}_{case.case_id}"
        case_out_dir.mkdir(parents=True, exist_ok=True)
        json_path = case_out_dir / f"{case.case_id}_sections.json"
        html_path = case_out_dir / f"{case.case_id}_sections.html"

        logger.info(f"  → 写入 JSON：{json_path.relative_to(PROJECT_ROOT)}")
        json_path.write_text(
            json.dumps(case.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Step C: 知识图谱构建
        logger.info("  → Step C: 知识图谱构建...")
        t_c = time.perf_counter()
        section_graphs = await _run_step_c_async(case, case_out_dir, logger)
        logger.info(
            f"  → Step C 完成：{len(section_graphs)} 个 SectionGraph，"
            f"耗时 {time.perf_counter() - t_c:.2f}s"
        )

        logger.info(f"  → 写入 HTML：{html_path.relative_to(PROJECT_ROOT)}")
        html_path.write_text(
            render_html(case, source_filename=txt_path.name, section_graphs=section_graphs),
            encoding="utf-8",
        )

        elapsed = time.perf_counter() - t0
        logger.info(f"  ✓ 完成，耗时 {elapsed:.2f}s  输出：{case_out_dir.relative_to(PROJECT_ROOT)}")

    total = time.perf_counter() - run_wall_start
    logger.info("=" * 60)
    logger.info(f"全部完成 {len(selected)} 个文件，总耗时 {total:.2f}s")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_SECTION_COLORS: dict[str, str] = {
    "general_info":          "#fde68a",
    "chief_complaint":       "#fca5a5",
    "present_illness":       "#bfdbfe",
    "past_medical_history":  "#fbcfe8",
    "medication_history":    "#fed7aa",
    "exposure_history":      "#ddd6fe",
    "allergy_history":       "#fef3c7",
    "family_history":        "#fde2e7",
    "physical_exam":         "#a7f3d0",
    "imaging_findings":      "#c7d2fe",
    "pulmonary_function":    "#bae6fd",
    "laboratory_findings":   "#d9f99d",
    "pathology_findings":    "#f5d0fe",
    "initial_diagnosis":     "#fdba74",
    "treatment_plan":        "#d8b4fe",
    "progress_note":         "#cbd5e1",
}


def _color_for(sid: str) -> str:
    return _SECTION_COLORS.get(sid, "#e5e7eb")


_CHUNK_TYPE_COLORS: dict[str, str] = {
    "temporal_episode":  "#bfdbfe",  # blue
    "exam_item":         "#d9f99d",  # green
    "medication_entry":  "#fed7aa",  # orange
    "procedure_entry":   "#e9d5ff",  # purple
    "exposure_entry":    "#fce7f3",  # pink
    "clinical_state":    "#fde68a",  # yellow
    "diagnosis_entry":   "#fca5a5",  # red
}

_CHUNK_TYPE_ZH: dict[str, str] = {
    "temporal_episode":  "时序叙事",
    "exam_item":         "检查项目",
    "medication_entry":  "药物记录",
    "procedure_entry":   "治疗操作",
    "exposure_entry":    "暴露/社会史",
    "clinical_state":    "临床状态",
    "diagnosis_entry":   "诊断陈述",
}


_NODE_TYPE_ZH: dict[str, str] = {
    "PATIENT": "患者主体",
    "SEX": "性别",
    "AGE": "年龄",
    "OCCUPATION": "职业",
    "BIRTHPLACE": "出生地/籍贯",
    "GENERAL_HEALTH_STATUS": "一般健康状态",
    "GENERAL_STATUS_ITEM": "一般状态项目",
    "GENERAL_STATUS_VALUE": "一般状态取值",
    "SYMPTOM": "症状",
    "SIGN": "体征",
    "VITAL_SIGN": "生命体征",
    "QUALIFIER": "修饰属性",
    "FUNCTIONAL_STATUS": "功能状态/活动耐量",
    "DISEASE": "疾病/诊断名",
    "DISEASE_SUBTYPE": "疾病亚型",
    "DIAGNOSIS_ASSERTION": "诊断陈述",
    "DIAGNOSIS_BASIS": "诊断依据",
    "ALLERGY": "过敏史/过敏原",
    "FAMILY_HISTORY_ITEM": "家族史项目",
    "PATHOGEN_OR_INFECTION": "病原体/感染事件",
    "TIME_REF": "时间参照点",
    "DURATION": "持续时长",
    "FREQUENCY": "频率",
    "CLINICAL_EVENT": "临床事件",
    "ENCOUNTER_LOCATION": "就诊地点/医疗机构",
    "TRIGGER": "诱发因素",
    "EXPOSURE": "暴露史实体",
    "EXPOSURE_AMOUNT": "暴露量",
    "EXPOSURE_DURATION": "暴露持续时间",
    "HABIT": "生活习惯",
    "PATHOGEN": "病原体",
    "VACCINATION": "疫苗接种",
    "DRUG": "药物名",
    "DRUG_DOSE": "药物剂量",
    "DRUG_ROUTE": "给药途径",
    "DRUG_FREQUENCY": "药物频率",
    "PROCEDURE": "手术/操作",
    "OXYGEN_THERAPY": "氧疗",
    "CLINICAL_TRIAL": "临床试验",
    "CONSULTATION": "会诊/MDT",
    "MEDICATION_ADHERENCE": "用药依从性",
    "TREATMENT_PLAN": "治疗方案",
    "TREATMENT_RESPONSE": "疗效结果",
    "EXAM_TEST": "检查项目",
    "LAB_TEST": "化验项目",
    "LAB_RESULT": "化验结果",
    "BLOOD_GAS_METRIC": "血气指标",
    "BLOOD_GAS_RESULT": "血气结果",
    "AUTOANTIBODY": "自身抗体",
    "IMAGING_FINDING": "影像所见",
    "IMAGING_PATTERN": "影像模式",
    "PFT_METRIC": "肺功能指标",
    "PFT_RESULT": "肺功能结果",
    "CARDIAC_FINDING": "心超/心脏所见",
    "ECG_FINDING": "心电图所见",
    "ULTRASOUND_FINDING": "超声所见",
    "PATHOLOGY_FINDING": "病理所见",
    "BODY_SITE": "解剖部位",
    "SEVERITY": "严重程度",
}


def _chunk_color(ctype: str) -> str:
    return _CHUNK_TYPE_COLORS.get(ctype, "#e5e7eb")


def render_html(
    case: DiscourseSectionedCase,
    source_filename: str,
    section_graphs: list[SectionGraph] | None = None,
) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{escape(case.case_id)} sections</title>
  <style>
    body {{ margin: 32px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            line-height: 1.7; color: #1f2937; background: #f8fafc; }}
    .node-type {{ background: #e0f2fe; padding: 1px 5px; border-radius: 3px; font-size: 11px; }}
    .edge-type {{ background: #fef3c7; padding: 1px 5px; border-radius: 3px; font-size: 11px; }}
    .negated {{ color: #dc2626; font-style: italic; }}
    .low-conf {{ color: #d97706; font-size: 11px; }}
    main {{ max-width: 1180px; margin: 0 auto; }}
    h1 {{ margin-bottom: 8px; font-size: 26px; }}
    h2 {{ margin-top: 36px; font-size: 20px; border-bottom: 2px solid #94a3b8; padding-bottom: 6px; }}
    h3.subhead {{ margin: 22px 0 10px; font-size: 15px; color: #334155; }}
    .meta {{ margin-bottom: 24px; color: #475569; font-size: 14px; }}
    .report-section {{ margin-top: 26px; }}
    .note {{ color: #475569; font-size: 13px; margin: 8px 0 14px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 20px; }}
    .legend span {{ padding: 3px 10px; border-radius: 4px; font-size: 12px; }}
    .pipeline {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; margin: 14px 0 18px; }}
    .stage-card {{ background: #ffffff; border: 1px solid #cbd5e1; border-radius: 5px; padding: 10px; min-height: 88px; }}
    .stage-card strong {{ display: block; font-size: 13px; margin-bottom: 4px; }}
    .stage-card .count {{ display: block; color: #0f172a; font-size: 18px; font-weight: 700; margin-bottom: 2px; }}
    .stage-card .desc {{ color: #64748b; font-size: 12px; line-height: 1.45; }}
    .type-dictionary {{ margin: 16px 0 24px; }}
    .type-dictionary details {{ background: #ffffff; border: 1px solid #cbd5e1; border-radius: 5px; margin: 8px 0; }}
    .type-dictionary summary {{ cursor: pointer; padding: 9px 12px; font-weight: 600; color: #334155; }}
    .type-dictionary .inner {{ padding: 0 12px 12px; }}
    .type-dictionary table {{ margin-top: 4px; }}
    .type-dictionary .unused {{ color: #94a3b8; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #e2e8f0; }}
    td.code {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; color: #475569; white-space: nowrap; }}
    tr td:first-child {{ width: 56px; text-align: center; white-space: nowrap; }}
    .section-card {{ margin: 14px 0; padding: 14px 16px; border-left: 6px solid; border-radius: 4px;
                     background: #ffffff; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
    .section-card h3 {{ margin: 0 0 8px; font-size: 16px; }}
    .section-card .fields {{ font-size: 12px; color: #475569; margin-bottom: 8px; }}
    .chunk-card {{ margin: 8px 0; padding: 10px 14px; border-left: 4px solid; border-radius: 3px;
                   background: #f8fafc; font-size: 13px; }}
    .chunk-card .chunk-meta {{ font-size: 11px; color: #6b7280; margin-bottom: 4px; }}
    .chunk-card .chunk-summary {{ font-style: italic; color: #374151; margin-bottom: 6px; font-size: 12px; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word;
           font-family: ui-monospace, Menlo, Consolas, monospace; background: #f1f5f9;
           padding: 10px; border-radius: 4px; font-size: 13px; }}
    .highlight {{ white-space: pre-wrap; word-break: break-word;
                  font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 13px;
                  background: #ffffff; padding: 14px; border: 1px solid #cbd5e1; border-radius: 4px; }}
    .highlight span {{ padding: 1px 0; border-radius: 2px; }}
    code {{ background: rgba(15,23,42,0.06); padding: 0 4px; border-radius: 3px; font-size: 12px; }}
    .changed {{ background: #fef08a !important; font-weight: bold; }}
    .stat-badge {{ display: inline-block; background: #e2e8f0; border-radius: 3px;
                   padding: 1px 7px; font-size: 12px; margin-right: 4px; }}
    .graph-view {{ border: 1px solid #e2e8f0; border-radius: 6px; overflow: hidden; margin: 14px 0; }}
    .graph-toolbar {{ display: flex; align-items: center; gap: 8px; padding: 8px 12px;
                      background: #f8fafc; border-bottom: 1px solid #e2e8f0; }}
    .graph-toolbar button {{ padding: 4px 12px; border: 1px solid #94a3b8; border-radius: 4px;
                             background: #fff; cursor: pointer; font-size: 12px; }}
    .graph-toolbar button:hover {{ background: #e2e8f0; }}
    .cy-graph {{ height: 480px; background: #fafafa; }}
    .cy-graph.is-empty {{ display: flex; align-items: center; justify-content: center;
                          color: #94a3b8; font-size: 14px; padding: 24px; }}
    .graph-legend {{ display: flex; flex-wrap: wrap; gap: 6px; padding: 8px 12px;
                     background: #f8fafc; border-top: 1px solid #e2e8f0; font-size: 11px; color: #475569; }}
    .graph-legend i {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px;
                       vertical-align: middle; margin-right: 3px; }}
    @media (max-width: 900px) {{
      body {{ margin: 18px; }}
      .pipeline {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>{escape(case.case_id)}</h1>
    <div class="meta">
      Source: {escape(source_filename)}<br>
      text_hash: <code>{escape(case.text_hash)}</code><br>
      <span class="stat-badge">原文 {len(case.raw_text)} 字符</span>
      <span class="stat-badge">句子 {len(case.sentences)}</span>
      <span class="stat-badge">篇章 section {len(case.sections)}</span>
      <span class="stat-badge">语义 chunk {len(case.chunks)}</span>
      <span class="stat-badge">审查修正 {sum(1 for lab in case.labels if lab.changed_in_review)} 处</span>
      <span class="stat-badge">图节点 {sum(len(g.nodes) for g in (section_graphs or []))}</span>
      <span class="stat-badge">图边 {sum(len(g.edges) for g in (section_graphs or []))}</span>
    </div>

    <h2 id="overview">Overview: Pipeline 层级</h2>
    {_render_pipeline_overview(case, section_graphs)}

    <h2 id="type-dictionary">Type Dictionary: 关键类型字典</h2>
    {_render_type_dictionary(case, section_graphs)}

    <section class="report-section" id="sectioning">
    <h2>A. Sectioning: 原文 → 句子 → 篇章 section</h2>
    <p class="note">这一组使用 <code>section_id</code> 标签，目标是判断病历文本属于哪个篇章类别。</p>
    <h3 class="subhead">Section taxonomy</h3>
    {_legend()}

    <h3 class="subhead">Reviewed section 标签着色</h3>
    {_render_highlight(case)}

    <h3 class="subhead">Phase 1: 句子切分 (Stanza)</h3>
    {_render_sentences_table(case)}

    <h3 class="subhead">Phase 2: 单句初判 / Phase 3: 全局审查</h3>
    {_render_labels_table(case)}

    <h3 class="subhead">Phase 3 output: 合并后的篇章 section</h3>
    {_render_sections(case)}
    </section>

    <section class="report-section" id="chunking">
    <h2>B. Chunking: 篇章 section → semantic chunk</h2>
    <p class="note">这一组使用 <code>chunk_type</code> 标签。chunk 是后续建图的最小语义单位，不等同于上面的 <code>section_id</code>。</p>
    {_render_chunks(case)}
    </section>

    <section class="report-section" id="graph">
    <h2>C. Graph Building: semantic chunk → SectionGraph</h2>
    {_render_graphs(section_graphs)}
    </section>
  </main>
<script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
<script src="https://unpkg.com/cytoscape-fcose@2.2.0/cytoscape-fcose.js"></script>
{_CYTOSCAPE_JS}
</body>
</html>
"""


def _render_pipeline_overview(
    case: DiscourseSectionedCase,
    section_graphs: list[SectionGraph] | None,
) -> str:
    changed = sum(1 for lab in case.labels if lab.changed_in_review)
    graph_count = len(section_graphs or [])
    node_count = sum(len(g.nodes) for g in (section_graphs or []))
    edge_count = sum(len(g.edges) for g in (section_graphs or []))
    cards = [
        ("Input", f"{len(case.raw_text)}", "原文字符"),
        ("Phase 1", f"{len(case.sentences)}", "Stanza 句子切分"),
        ("Phase 2-3", f"{len(case.sections)}", f"reviewed section；修正 {changed} 处"),
        ("Phase 4", f"{len(case.chunks)}", "semantic chunk；建图输入"),
        ("Step C", f"{graph_count}", f"SectionGraph；节点 {node_count} / 边 {edge_count}"),
    ]
    items = [
        "<div class='stage-card'>"
        f"<strong>{escape(title)}</strong>"
        f"<span class='count'>{escape(count)}</span>"
        f"<span class='desc'>{escape(desc)}</span>"
        "</div>"
        for title, count, desc in cards
    ]
    return (
        '<div class="pipeline">'
        + "".join(items)
        + "</div>"
        + '<p class="note">阅读顺序：先看 A 的篇章归类，再看 B 的 chunk 粒度，最后看 C 的图谱结果。</p>'
    )


def _render_type_dictionary(
    case: DiscourseSectionedCase,
    section_graphs: list[SectionGraph] | None,
) -> str:
    from collections import Counter

    section_counts = Counter(sec.section_id for sec in case.sections)
    chunk_counts = Counter(ch.chunk_type for ch in case.chunks)
    node_counts = Counter(
        node.node_type
        for graph in (section_graphs or [])
        for node in graph.nodes
    )
    edge_counts = Counter(
        edge.edge_type
        for graph in (section_graphs or [])
        for edge in graph.edges
    )

    section_rows = [
        _type_row(defn.zh_name, sid, section_counts.get(sid, 0))
        for sid, defn in DISCOURSE_TAXONOMY.items()
    ]
    chunk_rows = [
        _type_row(_CHUNK_TYPE_ZH.get(ctype, ctype), ctype, chunk_counts.get(ctype, 0))
        for ctype in _CHUNK_TYPE_COLORS
    ]
    node_type_codes = sorted(set(_NODE_TYPE_ZH) | set(node_counts))
    node_rows = [
        _type_row(_NODE_TYPE_ZH.get(code, code), code, node_counts.get(code, 0))
        for code in node_type_codes
    ]
    edge_type_codes = sorted(set(EDGE_TYPE_DESCRIPTIONS) | set(edge_counts))
    edge_rows = [
        _type_row(EDGE_TYPE_DESCRIPTIONS.get(code, code), code, edge_counts.get(code, 0))
        for code in edge_type_codes
    ]

    return (
        '<div class="type-dictionary">'
        '<p class="note">这里展示当前 pipeline 支持的关键类型，并标出本次输出中实际出现的次数。</p>'
        + _type_group("Section types / 篇章类型", section_rows, len(DISCOURSE_TAXONOMY), True)
        + _type_group("Chunk types / 语义块类型", chunk_rows, len(_CHUNK_TYPE_COLORS), False)
        + _type_group("Node types / 图节点类型", node_rows, len(node_type_codes), False)
        + _type_group("Relation types / 图关系类型", edge_rows, len(edge_type_codes), False)
        + "</div>"
    )


def _type_group(title: str, rows: list[str], total: int, open_by_default: bool) -> str:
    open_attr = " open" if open_by_default else ""
    return (
        f"<details{open_attr}>"
        f"<summary>{escape(title)}（{total} 类）</summary>"
        '<div class="inner">'
        "<table><thead><tr><th>中文名/说明</th><th>代码名称</th><th>当前数量</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "</div></details>"
    )


def _type_row(zh_name: str, code: str, count: int) -> str:
    row_class = " class='unused'" if count == 0 else ""
    return (
        f"<tr{row_class}>"
        f"<td>{escape(zh_name)}</td>"
        f"<td><code>{escape(code)}</code></td>"
        f"<td>{count}</td>"
        "</tr>"
    )


def _legend() -> str:
    items = []
    for sid, defn in DISCOURSE_TAXONOMY.items():
        color = _color_for(sid)
        items.append(
            f'<span style="background:{color}"><strong>{escape(defn.zh_name)}</strong> '
            f'<code>{escape(sid)}</code></span>'
        )
    return f'<div class="legend">{"".join(items)}</div>'


def _render_highlight(case: DiscourseSectionedCase) -> str:
    if not case.labels:
        return f'<pre class="highlight">{escape(case.raw_text)}</pre>'
    pieces: list[str] = []
    cursor = 0
    for lab in case.labels:
        s = lab.sentence
        if s.start_char > cursor:
            pieces.append(escape(case.raw_text[cursor:s.start_char]))
        color = _color_for(lab.reviewed_section_id)
        title = f"S{s.index} | {lab.reviewed_section_id}"
        if lab.changed_in_review:
            title += f" (was {lab.initial_section_id})"
        pieces.append(
            f'<span style="background:{color}" title="{escape(title)}">'
            f'{escape(case.raw_text[s.start_char:s.end_char])}</span>'
        )
        cursor = s.end_char
    if cursor < len(case.raw_text):
        pieces.append(escape(case.raw_text[cursor:]))
    return f'<div class="highlight">{"".join(pieces)}</div>'


def _render_sentences_table(case: DiscourseSectionedCase) -> str:
    rows = []
    for s in case.sentences:
        rows.append(
            f"<tr><td>S{s.index}</td>"
            f"<td class='code'>[{s.start_char}:{s.end_char}] len={s.end_char-s.start_char}</td>"
            f"<td>{escape(s.text)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>#</th><th>span</th><th>text</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_labels_table(case: DiscourseSectionedCase) -> str:
    rows = []
    for lab in case.labels:
        s = lab.sentence
        init_color = _color_for(lab.initial_section_id)
        rev_color = _color_for(lab.reviewed_section_id)
        rev_class = "changed" if lab.changed_in_review else ""
        rows.append(
            f"<tr><td>S{s.index}</td>"
            f"<td style='background:{init_color}'><code>{escape(lab.initial_section_id)}</code></td>"
            f"<td style='background:{rev_color}' class='{rev_class}'>"
            f"<code>{escape(lab.reviewed_section_id)}</code></td>"
            f"<td class='code'>{escape(lab.initial_reasoning[:80])}</td>"
            f"<td class='code'>{escape(lab.reviewed_reasoning[:80])}</td>"
            f"<td>{escape(s.text)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>#</th><th>initial</th><th>reviewed</th>"
        "<th>init reason</th><th>review reason</th><th>text</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_sections(case: DiscourseSectionedCase) -> str:
    blocks = []
    for sec in case.sections:
        color = _color_for(sec.section_id)
        ids = ", ".join(f"S{i}" for i in sec.sentence_indices)
        blocks.append(
            f'<article class="section-card" style="border-color:{color}">'
            f'  <h3>[{sec.index}] {escape(sec.section_name)} '
            f'<code style="background:{color}">{escape(sec.section_id)}</code></h3>'
            f'  <div class="fields">span [{sec.start_char}:{sec.end_char}] ｜ 句子 {ids}'
            f'  ｜ 长度 {sec.end_char - sec.start_char} 字符</div>'
            f'  <pre>{escape(sec.text)}</pre>'
            f'</article>'
        )
    return "\n".join(blocks)


def _render_chunks(case: DiscourseSectionedCase) -> str:
    if not case.chunks:
        return '<p style="color:#6b7280;">No chunks (Phase 4 not run or no sections).</p>'

    # chunk type legend
    legend_items = [
        f'<span style="background:{color};padding:3px 10px;border-radius:4px;'
        f'font-size:12px;"><strong>{escape(zh)}</strong> <code>{escape(ctype)}</code></span>'
        for ctype, color in _CHUNK_TYPE_COLORS.items()
        for zh in [_CHUNK_TYPE_ZH.get(ctype, ctype)]
    ]
    legend_html = (
        '<h3 class="subhead">Chunk type taxonomy</h3>'
        f'<div class="legend">{" ".join(legend_items)}</div>'
    )

    # full-text view colored by chunk_type
    chunks_sorted = sorted(case.chunks, key=lambda c: c.start_char)
    pieces: list[str] = []
    cursor = 0
    for chunk in chunks_sorted:
        if chunk.start_char > cursor:
            pieces.append(escape(case.raw_text[cursor:chunk.start_char]))
        color = _chunk_color(chunk.chunk_type)
        title = f"{chunk.chunk_id} | {chunk.chunk_type} | {chunk.standalone_summary}"
        pieces.append(
            f'<span style="background:{color}" title="{escape(title)}">'
            f'{escape(case.raw_text[chunk.start_char:chunk.end_char])}</span>'
        )
        cursor = chunk.end_char
    if cursor < len(case.raw_text):
        pieces.append(escape(case.raw_text[cursor:]))
    highlight_html = (
        '<h3 class="subhead">原文按 chunk_type 着色</h3>'
        f'<div class="highlight">{"".join(pieces)}</div>'
        '<h3 class="subhead">按 section 展开的 chunks</h3>'
    )

    # per-section detailed cards
    from itertools import groupby
    sections_by_index = {sec.index: sec for sec in case.sections}
    cards: list[str] = []
    for sec_idx, sec_chunks in groupby(chunks_sorted, key=lambda c: c.section_index):
        sec = sections_by_index.get(sec_idx)
        sec_color = _color_for(sec.section_id) if sec else "#e5e7eb"
        sec_label = f"[{sec_idx}] {escape(sec.section_name)} <code style='background:{sec_color}'>{escape(sec.section_id)}</code>" if sec else f"[{sec_idx}]"
        chunk_list = list(sec_chunks)
        inner_rows = []
        for ch in chunk_list:
            ccolor = _chunk_color(ch.chunk_type)
            czh = _CHUNK_TYPE_ZH.get(ch.chunk_type, ch.chunk_type)
            auto_fill = "⚠️自动填充" if "自动填充" in ch.standalone_summary else ""
            inner_rows.append(
                f'<div class="chunk-card" style="border-color:{ccolor}">'
                f'  <div class="chunk-meta">'
                f'    <code>{escape(ch.chunk_id)}</code>'
                f'    &nbsp;<span style="background:{ccolor};padding:1px 6px;border-radius:3px;font-size:11px;">{escape(czh)}</span>'
                f'    &nbsp;<code>[{ch.start_char}:{ch.end_char}]</code>'
                f'    {auto_fill}'
                f'  </div>'
                f'  <div class="chunk-summary">📌 {escape(ch.standalone_summary)}</div>'
                f'  <pre>{escape(ch.text)}</pre>'
                f'</div>'
            )
        cards.append(
            f'<article class="section-card" style="border-color:{sec_color}">'
            f'  <h3>{sec_label} &nbsp;<small style="font-weight:normal;color:#6b7280;">→ {len(chunk_list)} chunks</small></h3>'
            f'  {"".join(inner_rows)}'
            f'</article>'
        )

    return legend_html + highlight_html + "\n".join(cards)


def _render_graphs(section_graphs: list[SectionGraph] | None) -> str:
    if not section_graphs:
        return '<p style="color:#6b7280;">Step C 未运行或无图数据。</p>'

    cards: list[str] = []
    for sg in section_graphs:
        container_id = sg.graph_id.replace("_", "-")

        elements = _build_section_graph_elements(sg)
        payload = json.dumps(elements, ensure_ascii=False)
        payload_attr = (
            payload.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
        )

        toolbar = (
            "<div class='graph-toolbar'>"
            f"<button type='button' data-cy-action='fit' data-cy-target='{container_id}'>自适应</button>"
            f"<button type='button' data-cy-action='relayout' data-cy-target='{container_id}'>重新布局</button>"
            "<span style='color:#94a3b8;font-size:12px;'>滚轮缩放 · 拖拽节点 · 悬停查看详情</span>"
            "</div>"
        )
        graph_div = (
            f"<div class='cy-graph' id='{container_id}' "
            f"data-cy-elements=\"{payload_attr}\"></div>"
        )
        graph_view = f"<div class='graph-view'>{toolbar}{graph_div}{_graph_legend_html()}</div>"

        node_rows = []
        for node in sg.nodes:
            neg = '<span class="negated">[否]</span>' if node.negated else ""
            imp = '<span class="low-conf">[隐式]</span>' if node.implicit else ""
            node_rows.append(
                f"<tr><td class='code'>{escape(node.node_id[-10:])}</td>"
                f"<td><span class='node-type'>{escape(node.node_type)}</span></td>"
                f"<td>{escape(node.source_text)}{neg}{imp}</td>"
                f"<td class='code'>{escape(node.certainty)}</td></tr>"
            )
        node_table = (
            "<table><thead><tr><th>ID</th><th>类型</th><th>原文</th><th>确定性</th></tr></thead>"
            f"<tbody>{''.join(node_rows) or '<tr><td colspan=4 style=color:#6b7280>无节点</td></tr>'}</tbody></table>"
        )

        edge_rows = []
        for edge in sg.edges:
            lc = '<span class="low-conf">⚠</span>' if edge.low_confidence else ""
            edge_rows.append(
                f"<tr><td class='code'>{escape(edge.edge_id[-10:])}</td>"
                f"<td><span class='edge-type'>{escape(edge.edge_type)}</span></td>"
                f"<td class='code'>{escape(edge.source_node_id[-10:])}</td>"
                f"<td class='code'>{escape(edge.target_node_id[-10:])}</td>"
                f"<td>{escape(f'{edge.confidence:.2f}')}{lc}</td>"
                f"<td>{escape(edge.reasoning[:60])}</td></tr>"
            )
        edge_table = (
            "<table><thead><tr><th>ID</th><th>类型</th><th>源</th><th>目标</th>"
            "<th>置信度</th><th>推断依据（截断）</th></tr></thead>"
            f"<tbody>{''.join(edge_rows) or '<tr><td colspan=6 style=color:#6b7280>无边</td></tr>'}</tbody></table>"
        )

        frame_rows = []
        for frame in sg.frames:
            frame_rows.append(
                f"<tr><td class='code'>{escape(frame.frame_id[-18:])}</td>"
                f"<td>{escape(frame.frame_type)}</td>"
                f"<td>{escape(frame.source_text)}</td>"
                f"<td>{len(frame.node_ids)}</td><td>{len(frame.edge_ids)}</td></tr>"
            )
        frame_table = (
            "<table><thead><tr><th>ID</th><th>类型</th><th>上下文</th><th>节点</th><th>边</th></tr></thead>"
            f"<tbody>{''.join(frame_rows) or '<tr><td colspan=5 style=color:#6b7280>无上下文框</td></tr>'}</tbody></table>"
        )

        cards.append(
            f'<article class="section-card" style="border-color:#94a3b8">'
            f'  <h3>{escape(sg.section_name)} <code>{escape(sg.graph_id)}</code>'
            f'  &nbsp;<small style="font-weight:normal;color:#6b7280;">'
            f'  节点 {len(sg.nodes)} · 边 {len(sg.edges)}</small></h3>'
            f'  {graph_view}'
            f'  <details style="margin-top:10px"><summary style="cursor:pointer;font-size:13px;color:#475569;">'
            f'  节点列表（{len(sg.nodes)}）</summary>{node_table}</details>'
            f'  <details style="margin-top:6px"><summary style="cursor:pointer;font-size:13px;color:#475569;">'
            f'  边列表（{len(sg.edges)}）</summary>{edge_table}</details>'
            f'  <details style="margin-top:6px"><summary style="cursor:pointer;font-size:13px;color:#475569;">'
            f'  上下文框（{len(sg.frames)}）</summary>{frame_table}</details>'
            f'</article>'
        )
    return "\n".join(cards)


_NODE_TYPE_CATEGORY: dict[str, str] = {
    "PATIENT": "patient",
    "SEX": "patient", "AGE": "patient", "OCCUPATION": "patient",
    "BIRTHPLACE": "patient", "GENERAL_HEALTH_STATUS": "patient",
    "GENERAL_STATUS_ITEM": "patient", "GENERAL_STATUS_VALUE": "qualifier",
    "SYMPTOM": "symptom", "SIGN": "symptom", "VITAL_SIGN": "symptom",
    "QUALIFIER": "qualifier", "FUNCTIONAL_STATUS": "qualifier",
    "DISEASE": "disease", "DISEASE_SUBTYPE": "disease",
    "DIAGNOSIS_ASSERTION": "diagnosis", "DIAGNOSIS_BASIS": "diagnosis",
    "ALLERGY": "diagnosis", "FAMILY_HISTORY_ITEM": "diagnosis",
    "PATHOGEN_OR_INFECTION": "diagnosis",
    "TIME_REF": "time", "DURATION": "time", "FREQUENCY": "time",
    "CLINICAL_EVENT": "time", "ENCOUNTER_LOCATION": "time",
    "TRIGGER": "exposure", "EXPOSURE": "exposure", "PATHOGEN": "exposure",
    "VACCINATION": "exposure", "EXPOSURE_AMOUNT": "exposure",
    "EXPOSURE_DURATION": "exposure", "HABIT": "exposure",
    "DRUG": "treatment", "DRUG_DOSE": "treatment", "DRUG_ROUTE": "treatment",
    "DRUG_FREQUENCY": "treatment", "PROCEDURE": "treatment",
    "OXYGEN_THERAPY": "treatment", "CLINICAL_TRIAL": "treatment",
    "CONSULTATION": "treatment", "MEDICATION_ADHERENCE": "treatment",
    "TREATMENT_PLAN": "treatment", "TREATMENT_RESPONSE": "treatment",
    "EXAM_TEST": "lab", "LAB_TEST": "lab", "LAB_RESULT": "lab",
    "BLOOD_GAS_METRIC": "lab", "BLOOD_GAS_RESULT": "lab", "AUTOANTIBODY": "lab",
    "IMAGING_FINDING": "imaging", "IMAGING_PATTERN": "imaging",
    "PFT_METRIC": "pft", "PFT_RESULT": "pft",
    "CARDIAC_FINDING": "imaging", "ECG_FINDING": "imaging",
    "ULTRASOUND_FINDING": "imaging",
    "PATHOLOGY_FINDING": "pathology",
    "BODY_SITE": "bodysite",
    "SEVERITY": "severity",
}


def _build_section_graph_elements(sg: SectionGraph) -> list[dict]:
    elements: list[dict] = []
    for node in sg.nodes:
        label_text = "患者" if node.implicit and node.node_type == "PATIENT" else node.source_text
        label = label_text[:22] + ("…" if len(label_text) > 22 else "")
        if node.negated:
            label = "¬ " + label
        category = _NODE_TYPE_CATEGORY.get(node.node_type, "unknown")
        classes = f"node ntype-{category}"
        if node.negated:
            classes += " negated"
        if node.certainty in ("suspected", "uncertain"):
            classes += " uncertain"
        elif node.certainty == "excluded":
            classes += " excluded"
        elements.append({
            "group": "nodes",
            "data": {
                "id": node.node_id,
                "label": label,
                "node_type": node.node_type,
                "title": (
                    f"{node.node_type}\n"
                    f"{node.source_text}\n"
                    f"certainty: {node.certainty}\n"
                    f"negated: {'是' if node.negated else '否'}\n"
                    f"implicit: {'是' if node.implicit else '否'}\n"
                    f"{node.node_id}"
                ),
            },
            "classes": classes,
        })
    for edge in sg.edges:
        label = edge.edge_type
        if edge.negated:
            label = "¬ " + label
        elif edge.low_confidence:
            label = "? " + label
        classes = "edge"
        if edge.negated:
            classes += " negated"
        if edge.low_confidence:
            classes += " low-conf"
        elements.append({
            "group": "edges",
            "data": {
                "id": edge.edge_id,
                "source": edge.source_node_id,
                "target": edge.target_node_id,
                "label": label,
                "title": (
                    f"{edge.edge_type}\n"
                    f"confidence: {edge.confidence:.2f}\n"
                    f"reasoning: {edge.reasoning}\n"
                    f"negated: {'是' if edge.negated else '否'}\n"
                    f"{edge.edge_id}"
                ),
            },
            "classes": classes,
        })
    return elements


def _graph_legend_html() -> str:
    categories = [
        ("#dbeafe", "#3b82f6", "患者"),
        ("#dcfce7", "#16a34a", "症状/体征"),
        ("#fee2e2", "#ef4444", "诊断"),
        ("#ede9fe", "#8b5cf6", "治疗"),
        ("#e0f2fe", "#0ea5e9", "化验"),
        ("#e0e7ff", "#6366f1", "影像"),
        ("#fef9c3", "#facc15", "时间"),
        ("#fef3c7", "#d97706", "暴露史"),
        ("#cffafe", "#06b6d4", "肺功能"),
        ("#ffedd5", "#f97316", "病理"),
        ("#f3f4f6", "#6b7280", "解剖部位"),
        ("#fce7f3", "#ec4899", "严重程度"),
    ]
    items = [
        f"<span><i style='background:{bg};border:1.5px solid {bd};'></i>{escape(lbl)}</span>"
        for bg, bd, lbl in categories
    ]
    items.append("<span><i style='background:#f1f5f9;border:2px dashed #dc2626;'></i>否定</span>")
    items.append("<span><i style='background:#f1f5f9;border:1.5px dashed #94a3b8;'></i>疑诊/不确定</span>")
    return f"<div class='graph-legend'>{''.join(items)}</div>"


_CYTOSCAPE_JS = """<script>
(function () {
  if (typeof cytoscapeFcose !== "undefined") { cytoscape.use(cytoscapeFcose); }
  if (typeof cytoscape === "undefined") {
    document.querySelectorAll(".cy-graph").forEach(function (el) {
      el.classList.add("is-empty");
      el.textContent = "Cytoscape.js 加载失败（可能离线），无法渲染。表格见下方。";
    });
    return;
  }
  var instances = {};
  var graphStyle = [
    { selector: "node", style: { "background-color": "#f1f5f9", "border-color": "#94a3b8", "border-width": 1.5, "shape": "round-rectangle", "label": "data(label)", "color": "#1f2937", "font-size": 11, "text-wrap": "wrap", "text-max-width": 120, "text-valign": "center", "text-halign": "center", "padding": 8, "width": "label", "height": "label", "min-zoomed-font-size": 6 } },
    { selector: "node.ntype-patient",   style: { "background-color": "#dbeafe", "border-color": "#3b82f6", "border-width": 2 } },
    { selector: "node.ntype-symptom",   style: { "background-color": "#dcfce7", "border-color": "#16a34a" } },
    { selector: "node.ntype-qualifier", style: { "background-color": "#f0fdf4", "border-color": "#86efac" } },
    { selector: "node.ntype-disease",   style: { "background-color": "#fee2e2", "border-color": "#ef4444", "border-width": 2 } },
    { selector: "node.ntype-diagnosis", style: { "background-color": "#fef2f2", "border-color": "#fca5a5" } },
    { selector: "node.ntype-time",      style: { "background-color": "#fef9c3", "border-color": "#facc15" } },
    { selector: "node.ntype-exposure",  style: { "background-color": "#fef3c7", "border-color": "#d97706" } },
    { selector: "node.ntype-treatment", style: { "background-color": "#ede9fe", "border-color": "#8b5cf6" } },
    { selector: "node.ntype-lab",       style: { "background-color": "#e0f2fe", "border-color": "#0ea5e9" } },
    { selector: "node.ntype-imaging",   style: { "background-color": "#e0e7ff", "border-color": "#6366f1" } },
    { selector: "node.ntype-pft",       style: { "background-color": "#cffafe", "border-color": "#06b6d4" } },
    { selector: "node.ntype-pathology", style: { "background-color": "#ffedd5", "border-color": "#f97316" } },
    { selector: "node.ntype-bodysite",  style: { "background-color": "#f3f4f6", "border-color": "#6b7280" } },
    { selector: "node.ntype-severity",  style: { "background-color": "#fce7f3", "border-color": "#ec4899" } },
    { selector: "node.negated",  style: { "border-style": "dashed", "border-color": "#dc2626", "border-width": 2, "opacity": 0.8 } },
    { selector: "node.uncertain", style: { "border-style": "dashed", "opacity": 0.85 } },
    { selector: "node.excluded",  style: { "opacity": 0.55, "border-style": "dotted" } },
    { selector: "edge", style: { "curve-style": "bezier", "width": 1.4, "line-color": "#64748b", "target-arrow-color": "#64748b", "target-arrow-shape": "triangle", "arrow-scale": 0.9, "label": "data(label)", "font-size": 9, "color": "#334155", "text-background-color": "#ffffff", "text-background-opacity": 0.85, "text-background-padding": 2, "text-background-shape": "round-rectangle", "text-rotation": "autorotate", "min-zoomed-font-size": 5 } },
    { selector: "edge.negated",  style: { "line-style": "dashed", "line-color": "#dc2626", "target-arrow-color": "#dc2626", "color": "#b91c1c" } },
    { selector: "edge.low-conf", style: { "line-style": "dotted", "line-color": "#d97706", "target-arrow-color": "#d97706", "color": "#b45309" } },
    { selector: ":selected", style: { "border-color": "#2563eb", "border-width": 2.5, "line-color": "#2563eb", "target-arrow-color": "#2563eb" } },
  ];
  var hasFcose = (typeof cytoscapeFcose !== "undefined");
  function makeLayout(nodeCount) {
    if (hasFcose) {
      var edgeLen = nodeCount > 20 ? 50 : nodeCount > 10 ? 65 : 80;
      return { name: "fcose", animate: false, padding: 28, idealEdgeLength: edgeLen, edgeElasticity: 0.40, gravity: 0.35, gravityRange: 3.8, nodeSeparation: 60, packComponents: true, quality: "default", numIter: 2500, fit: true, randomize: true };
    }
    var repulsion = nodeCount > 20 ? 3500 : nodeCount > 10 ? 2800 : 1800;
    return { name: "cose", animate: false, padding: 28, nodeRepulsion: function () { return repulsion; }, idealEdgeLength: function () { return 65; }, edgeElasticity: function () { return 60; }, gravity: 0.8, numIter: 1000, nestingFactor: 1.2, componentSpacing: 40, fit: true };
  }
  function initContainer(el) {
    var raw = el.getAttribute("data-cy-elements");
    if (!raw) { return; }
    var elements;
    try { elements = JSON.parse(raw); }
    catch (err) { el.classList.add("is-empty"); el.textContent = "图数据解析失败：" + err.message; return; }
    if (!elements || !elements.length) { el.classList.add("is-empty"); el.textContent = "无图元素。"; return; }
    el.removeAttribute("data-cy-elements");
    var nodeCount = elements.filter(function (e) { return e.group === "nodes"; }).length;
    var cy = cytoscape({ container: el, elements: elements, style: graphStyle, wheelSensitivity: 0.25, minZoom: 0.05, maxZoom: 4.0 });
    var layout = cy.layout(makeLayout(nodeCount));
    layout.one("layoutstop", function () { cy.fit(undefined, 28); });
    layout.run();
    cy.on("mouseover", "node, edge", function (evt) { var t = evt.target.data("title"); if (t) { el.title = t; } });
    cy.on("mouseout", "node, edge", function () { el.title = ""; });
    instances[el.id] = cy;
  }
  function lazyInit() {
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) { initContainer(entry.target); observer.unobserve(entry.target); }
      });
    }, { rootMargin: "200px" });
    document.querySelectorAll(".cy-graph").forEach(function (el) { observer.observe(el); });
  }
  if ("IntersectionObserver" in window) { lazyInit(); }
  else { document.querySelectorAll(".cy-graph").forEach(initContainer); }
  document.addEventListener("click", function (event) {
    var btn = event.target.closest("[data-cy-action]");
    if (!btn) { return; }
    var targetId = btn.getAttribute("data-cy-target");
    var cy = instances[targetId];
    if (!cy) { var el = document.getElementById(targetId); if (el) { initContainer(el); cy = instances[targetId]; } }
    if (!cy) { return; }
    var action = btn.getAttribute("data-cy-action");
    if (action === "fit") { cy.fit(undefined, 28); }
    else if (action === "relayout") { var nc = cy.nodes().length; var lo = cy.layout(makeLayout(nc)); lo.one("layoutstop", function () { cy.fit(undefined, 28); }); lo.run(); }
  });
})();
</script>"""


def _resolve_project_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
