"""Runner for the all-LLM discourse sectionizer pipeline.

输出：
  outputs/runs/{run_id}_{case_id}/
    ├── {case_id}_sections.json     # 全 phase 数据
    └── {case_id}_sections.html     # 三个 phase 的可视化（含 initial vs reviewed diff）
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
from src.schemas.context_chunker.discourse_taxonomy import DISCOURSE_TAXONOMY  # noqa: E402


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
        logger.info(f"  → 写入 HTML：{html_path.relative_to(PROJECT_ROOT)}")
        html_path.write_text(
            render_html(case, source_filename=txt_path.name), encoding="utf-8"
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


def render_html(case: DiscourseSectionedCase, source_filename: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{escape(case.case_id)} sections</title>
  <style>
    body {{ margin: 32px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            line-height: 1.7; color: #1f2937; background: #f8fafc; }}
    main {{ max-width: 1180px; margin: 0 auto; }}
    h1 {{ margin-bottom: 8px; font-size: 26px; }}
    h2 {{ margin-top: 36px; font-size: 20px; border-bottom: 2px solid #94a3b8; padding-bottom: 6px; }}
    .meta {{ margin-bottom: 24px; color: #475569; font-size: 14px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 20px; }}
    .legend span {{ padding: 3px 10px; border-radius: 4px; font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #e2e8f0; }}
    td.code {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; color: #475569; white-space: nowrap; }}
    tr td:first-child {{ width: 56px; text-align: center; white-space: nowrap; }}
    .section-card {{ margin: 14px 0; padding: 14px 16px; border-left: 6px solid; border-radius: 4px;
                     background: #ffffff; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
    .section-card h3 {{ margin: 0 0 8px; font-size: 16px; }}
    .section-card .fields {{ font-size: 12px; color: #475569; margin-bottom: 8px; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word;
           font-family: ui-monospace, Menlo, Consolas, monospace; background: #f1f5f9;
           padding: 10px; border-radius: 4px; font-size: 13px; }}
    .highlight {{ white-space: pre-wrap; word-break: break-word;
                  font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 13px;
                  background: #ffffff; padding: 14px; border: 1px solid #cbd5e1; border-radius: 4px; }}
    .highlight span {{ padding: 1px 0; border-radius: 2px; }}
    code {{ background: rgba(15,23,42,0.06); padding: 0 4px; border-radius: 3px; font-size: 12px; }}
    .changed {{ background: #fef08a !important; font-weight: bold; }}
  </style>
</head>
<body>
  <main>
    <h1>{escape(case.case_id)}</h1>
    <div class="meta">
      Source: {escape(source_filename)}<br>
      text_hash: <code>{escape(case.text_hash)}</code><br>
      原文 {len(case.raw_text)} 字符 ｜
      句子 {len(case.sentences)} ｜
      篇章 section {len(case.sections)} ｜
      审查修正 {sum(1 for lab in case.labels if lab.changed_in_review)} 处
    </div>

    {_legend()}

    <h2>原文按 reviewed 标签着色</h2>
    {_render_highlight(case)}

    <h2>Phase 1: 句子切分 (Stanza)</h2>
    {_render_sentences_table(case)}

    <h2>Phase 2 &amp; 3: 句子分类（initial vs reviewed）</h2>
    {_render_labels_table(case)}

    <h2>Phase 3: 合并后的篇章 section</h2>
    {_render_sections(case)}
  </main>
</body>
</html>
"""


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


def _resolve_project_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
