#!/usr/bin/env python3
"""
04_extract_notes.py
===================
从 html/note/ 目录下的 HTML 文件提取注释术语表，
输出结构化数据至同目录下的 notes.csv 和 notes.jsonl。
遇到无法解析的条目，写入 html/note/report.md 报告。

用法：
    python src/04_extract_notes.py

每条记录字段：
    id         - 注释编号（整数，来自 div id=divNNN）
    pali_word  - 巴利文词汇
    meaning    - 主要汉文释义
    meaning2   - 次要汉文释义（可为 null）
    note       - 完整注释文本
"""

import csv
import json
import logging
import re
import sys
from pathlib import Path

# ── 日志配置 ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent
NOTE_DIR = ROOT / "html" / "note"

# ── 正则 ──────────────────────────────────────────────────────────────────────

# 匹配 <div id=divNNN> ... </div>（单行，允许 id 值有无引号）
RE_DIV = re.compile(
    r'<div\s+id=["\']?div(\d+)["\']?\s*>(.*?)</div>',
    re.IGNORECASE,
)

# 原有模式A：南傳作「meaning」(pali_word，另譯為「meaning2」)
RE_PATTERN_A = re.compile(
    r'南傳作「([^」]+)」\s*\(([^，,）)]+)[，,]'   # meaning + pali_word
    r'(?:.*?另譯為「([^」]+)」)?',                  # meaning2（可选）
)

# 原有模式B：「meaning」(pali_word)  无另译
RE_PATTERN_B = re.compile(
    r'「([^」]+)」\s*\(([a-zA-ZāīūṃṅñṭḍḷṇśṣḥĀĪŪṂṄÑṬḌḶṆŚṢḤ][^\)]*)\)'
)

# 新增模式C：「meaning」(pali_word，另譯為「meaning2」)  无「南傳作」
RE_PATTERN_C = re.compile(
    r'「([^」]+)」\s*\(([a-zA-ZāīūṃṅñṭḍḷṇśṣḥĀĪŪṂṄÑṬḌḶṆŚṢḤ][^，,)]*)[，,]\s*另譯為「([^」]+)」\)'
)


def extract_fields(note_text: str) -> dict:
    """
    从注释原文中提取 pali_word、meaning、meaning2。
    优先级：模式A（南傳作）> 模式C（无南傳作但有另译）> 模式B（无另译）
    """
    # 模式 A
    m = RE_PATTERN_A.search(note_text)
    if m:
        meaning = m.group(1).strip() or None
        pali_word = m.group(2).strip() or None
        meaning2 = m.group(3).strip() if m.group(3) else None
        return {"pali_word": pali_word.lower(), "meaning": meaning, "meaning2": meaning2}

    # 模式 C（无「南傳作」，但括号内有「另譯為」）
    m = RE_PATTERN_C.search(note_text)
    if m:
        meaning = m.group(1).strip() or None
        pali_word = m.group(2).strip() or None
        meaning2 = m.group(3).strip() or None
        return {"pali_word": pali_word.lower(), "meaning": meaning, "meaning2": meaning2}

    # 模式 B（仅 meaning + pali_word，无另译）
    m = RE_PATTERN_B.search(note_text)
    if m:
        meaning = m.group(1).strip() or None
        pali_word = m.group(2).strip() or None
        return {"pali_word": pali_word.lower(), "meaning": meaning, "meaning2": None}

    return {"pali_word": None, "meaning": None, "meaning2": None}


def parse_html_file(html_path: Path) -> tuple[list[dict], list[dict]]:
    """
    解析单个 HTML 文件，返回 (records, failures)。
    records  - 成功解析的条目列表
    failures - 解析失败或字段缺失的条目列表（含原始 note 文本）
    """
    records:  list[dict] = []
    failures: list[dict] = []

    text = html_path.read_text(encoding="utf-8")

    for m in RE_DIV.finditer(text):
        div_id    = int(m.group(1))
        note_text = m.group(2).strip()

        # 去除内嵌 HTML 标签（保留文本）
        note_clean = re.sub(r"<[^>]+>", "", note_text).strip()

        fields = extract_fields(note_clean)

        record = {
            "id":        div_id,
            "pali_word": fields["pali_word"],
            "meaning":   fields["meaning"],
            "meaning2":  fields["meaning2"],
            "note":      note_clean,
            "source":    html_path.name,
        }

        records.append(record)

        # 记录字段缺失情况
        missing = [k for k in ("pali_word", "meaning") if fields[k] is None]
        if missing:
            failures.append({
                "source":  html_path.name,
                "id":      div_id,
                "missing": missing,
                "note":    note_clean[:120],
            })

    return records, failures


def write_csv(records: list[dict], out_path: Path) -> None:
    """写入 CSV，字段顺序固定，None 输出为空字符串。"""
    fieldnames = ["id", "pali_word", "meaning", "meaning2", "note", "source"]
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        # utf-8-sig 带 BOM，Excel 直接打开不乱码
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: ("" if rec[k] is None else rec[k]) for k in fieldnames})
    log.info("CSV 已写入 → %s（%d 条）", out_path, len(records))


def write_jsonl(records: list[dict], out_path: Path) -> None:
    """写入 JSONL，每行一条，None 保留为 JSON null。"""
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("JSONL 已写入 → %s（%d 条）", out_path, len(records))


def write_report(failures: list[dict], report_path: Path, total: int) -> None:
    """生成 Markdown 格式的解析报告。"""
    lines: list[str] = []
    lines.append("# 注释提取报告\n")
    lines.append(f"- 总条目数：{total}")
    lines.append(f"- 字段缺失条目：{len(failures)}\n")

    if not failures:
        lines.append("✅ 全部条目均成功提取所有字段。")
    else:
        lines.append("## 字段缺失详情\n")
        lines.append("| 来源文件 | id | 缺失字段 | 注释原文（前120字） |")
        lines.append("|----------|----|----------|---------------------|")
        for f in failures:
            missing_str = ", ".join(f["missing"])
            note_preview = f["note"].replace("|", "｜").replace("\n", " ")
            lines.append(
                f"| {f['source']} | {f['id']} | `{missing_str}` | {note_preview} |"
            )

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("报告已写入 → %s", report_path)


def main() -> None:
    if not NOTE_DIR.exists():
        log.error("目录不存在: %s", NOTE_DIR)
        sys.exit(1)

    html_files = sorted(NOTE_DIR.glob("*.html"))
    if not html_files:
        log.error("html/note/ 下没有 .html 文件")
        sys.exit(1)

    log.info("找到 %d 个 HTML 文件", len(html_files))

    all_records:  list[dict] = []
    all_failures: list[dict] = []

    for html_path in html_files:
        log.info("══ 解析 %s ══", html_path.name)
        records, failures = parse_html_file(html_path)
        log.info("  提取 %d 条，字段缺失 %d 条", len(records), len(failures))
        all_records.extend(records)
        all_failures.extend(failures)

    # 按 id 排序
    all_records.sort(key=lambda r: r["id"])

    # 输出
    write_csv(all_records,  NOTE_DIR / "notes.csv")
    write_jsonl(all_records, NOTE_DIR / "notes.jsonl")
    write_report(all_failures, NOTE_DIR / "report.md", total=len(all_records))

    log.info("完成：共 %d 条，字段缺失 %d 条", len(all_records), len(all_failures))


if __name__ == "__main__":
    main()