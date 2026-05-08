#!/usr/bin/env python3
"""
02_align.py
===========
从 html/{corpus}/para_map.jsonl 读取 HTML↔段落对照表，
查询对应的 pali_sentences，使用 LLM 将中文译文对齐至各句，
结果写入 jsonl/{corpus}/{html_stem}.jsonl。
同时生成 jsonl/{corpus}/{html_stem}.md 对齐质量报告。

前置步骤：
    1. python src/01_download.py --corpus milinda --end 36
    2. python src/03_scan_paragraphs.py --corpus milinda

用法：
    python src/02_align.py --corpus milinda
    python src/02_align.py --corpus milinda --start 1 --end 3
"""

import argparse
import difflib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import psycopg2
import psycopg2.extras
import toml
from dotenv import load_dotenv

from llm_align import LlmUsage, llm_split

# ── 日志配置 ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


# ═════════════════════════════════════════════════════════════════════════════
# HTML 解析
# ═════════════════════════════════════════════════════════════════════════════

class DivExtractor(HTMLParser):
    """提取指定 id 的 div 内部文本，<br> 转换为换行。"""

    def __init__(self, target_id: str):
        super().__init__()
        self.target_id = target_id
        self.depth = 0
        self.in_target = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        if self.in_target:
            self.depth += 1
            if tag == "br":
                self.parts.append("\n")
        else:
            if dict(attrs).get("id") == self.target_id:
                self.in_target = True
                self.depth = 1

    def handle_endtag(self, tag: str):
        if self.in_target:
            self.depth -= 1
            if self.depth == 0:
                self.in_target = False

    def handle_data(self, data: str):
        if self.in_target:
            self.parts.append(data)

    @property
    def text(self) -> str:
        return "".join(self.parts)


def extract_text(html: str, div_id: str) -> str:
    p = DivExtractor(div_id)
    p.feed(html)
    return p.text


# ═════════════════════════════════════════════════════════════════════════════
# 文本正规化（用于 diff 比对）
# ═════════════════════════════════════════════════════════════════════════════

def normalize_for_diff(text: str) -> str:
    """
    去除空白、全角空格、换行，用于比对原文与拼接后文本是否一致。
    比对时不关心空白差异，只关心实质内容。
    """
    return re.sub(r"[\s\u3000]+", "", text)


# ═════════════════════════════════════════════════════════════════════════════
# 数据库
# ═════════════════════════════════════════════════════════════════════════════

def connect_db() -> psycopg2.extensions.connection:
    load_dotenv(ROOT / ".env")
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_DATABASE", "wikipali"),
        user=os.getenv("DB_USERNAME", "www"),
        password=os.getenv("DB_PASSWORD", ""),
    )


def load_pali_sentences(conn, book_id: int, paragraphs: list[int]) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, paragraph, word_begin, word_end, text
            FROM pali_sentences
            WHERE book = %s AND paragraph = ANY(%s)
            ORDER BY paragraph, word_begin
            """,
            (book_id, paragraphs),
        )
        return [dict(r) for r in cur.fetchall()]


# ═════════════════════════════════════════════════════════════════════════════
# 对照表加载
# ═════════════════════════════════════════════════════════════════════════════

def load_para_map(map_path: Path) -> dict[str, list[int]]:
    if not map_path.exists():
        return {}
    result = {}
    with map_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                result[rec["html"]] = list(range(rec["para_start"], rec["para_end"] + 1))
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Diff 比对
# ═════════════════════════════════════════════════════════════════════════════

def compute_diff(original: str, aligned: str) -> list[str]:
    """
    将 original 与 aligned 正规化后做字符级 diff。
    返回差异描述行列表；若无差异则返回空列表。
    对于看起来相同但 diff 仍报差异的区段，附加 Unicode 码位信息供排查。
    """
    orig_norm    = normalize_for_diff(original)
    aligned_norm = normalize_for_diff(aligned)

    if orig_norm == aligned_norm:
        return []

    matcher = difflib.SequenceMatcher(None, orig_norm, aligned_norm, autojunk=False)
    diffs: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        orig_chunk    = orig_norm[i1:i2]
        aligned_chunk = aligned_norm[j1:j2]

        if tag == "replace":
            diffs.append(f"- 原文: `{orig_chunk}`")
            diffs.append(f"+ 对齐: `{aligned_chunk}`")
        elif tag == "delete":
            diffs.append(f"- 原文多出: `{orig_chunk}`")
        elif tag == "insert":
            diffs.append(f"+ 对齐多出: `{aligned_chunk}`")

        # 若两段字符串表面相同但仍触发 diff，附加 Unicode 码位帮助排查
        # （常见原因：全角/半角混用、不同引号形式、零宽字符等）
        if orig_chunk == aligned_chunk and orig_chunk:
            codepoints = "  ".join(
                f"U+{ord(c):04X}({c!r})" for c in orig_chunk[:8]
            )
            diffs.append(f"  ℹ️ 字面相同但码位不同，原文码位: {codepoints}")
        elif orig_chunk and aligned_chunk:
            # 逐字符对比，找出第一个不同的码位
            for k, (a, b) in enumerate(zip(orig_chunk, aligned_chunk)):
                if a != b:
                    diffs.append(
                        f"  ℹ️ 首个不同字符位置 {k}: "
                        f"原文 U+{ord(a):04X}({a!r}) vs 对齐 U+{ord(b):04X}({b!r})"
                    )
                    break

    return diffs


# ═════════════════════════════════════════════════════════════════════════════
# Markdown 报告
# ═════════════════════════════════════════════════════════════════════════════

def write_report(
    report_path: Path,
    html_name:   str,
    usage:       LlmUsage,
    generated_at: str,
    diff_lines:  list[str],
    total_sentences: int,
    null_count:  int,
) -> None:
    """生成对齐质量 Markdown 报告。"""

    lines: list[str] = []
    lines.append(f"# 对齐报告：{html_name}\n")

    # ── 基本信息 ──────────────────────────────────────────────────────────────
    lines.append("## 基本信息\n")
    lines.append(f"| 项目 | 值 |")
    lines.append(f"|------|----|")
    lines.append(f"| HTML 文件 | `{html_name}` |")
    lines.append(f"| 生成时间 | {generated_at} |")
    lines.append(f"| 模型 | `{usage.model}` |")
    lines.append(f"| 句子总数 | {total_sentences} |")
    lines.append(f"| 未对齐（null）| {null_count} |")
    lines.append("")

    # ── Token 用量 ────────────────────────────────────────────────────────────
    lines.append("## Token 用量\n")
    lines.append(f"| 类型 | 数量 |")
    lines.append(f"|------|------|")
    lines.append(f"| Prompt tokens | {usage.prompt_tokens} |")
    lines.append(f"| Completion tokens | {usage.completion_tokens} |")
    lines.append(f"| Total tokens | {usage.total_tokens} |")
    if usage.extra:
        for k, v in usage.extra.items():
            lines.append(f"| {k} | {v} |")
    lines.append("")

    # ── 文本比对结果 ──────────────────────────────────────────────────────────
    lines.append("## 文本比对结果\n")
    if not diff_lines:
        lines.append("✅ 无差异：对齐后拼接文本与原文完全一致（忽略空白）。")
    else:
        lines.append(f"⚠️ 发现 {len(diff_lines)} 处差异（已忽略空白）：\n")
        lines.append("```diff")
        lines.extend(diff_lines)
        lines.append("```")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("  报告已写入 → %s", report_path.name)


# ═════════════════════════════════════════════════════════════════════════════
# 核心处理
# ═════════════════════════════════════════════════════════════════════════════

def process_chapter(
    conn,
    book_id:       int,
    html_path:     Path,
    paragraphs:    list[int],
    chapter_label: str,
    report_dir:    Path,
) -> list[dict]:
    """
    处理单个章节：
    1. 查询 pali_sentences
    2. LLM 对齐中文
    3. 拼接对齐结果与原文做 diff
    4. 生成 markdown 报告
    返回结果列表。
    """
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    html          = html_path.read_text(encoding="utf-8")
    chinese_clean = extract_text(html, "center")

    if not chinese_clean.strip():
        raise ValueError(f"{chapter_label}: #center 为空")

    # 查询 pali_sentences
    sentences = load_pali_sentences(conn, book_id, paragraphs)
    if not sentences:
        raise ValueError(f"{chapter_label}: pali_sentences 为空，段落={paragraphs}")

    log.info("  pali_sentences 共 %d 条", len(sentences))

    # LLM 对齐
    chinese_groups, usage = llm_split(chinese_clean, sentences)

    # ── Diff 比对 ──────────────────────────────────────────────────────────────
    # 将对齐结果拼接（跳过 None），与原始中文比对
    aligned_text = "".join(g for g in chinese_groups if g is not None)
    diff_lines   = compute_diff(chinese_clean, aligned_text)

    if diff_lines:
        log.warning("  文本比对：发现 %d 处差异", len(diff_lines))
    else:
        log.info("  文本比对：无差异 ✅")

    # ── 生成 markdown 报告 ─────────────────────────────────────────────────────
    null_count  = sum(1 for g in chinese_groups if g is None)
    report_path = report_dir / html_path.with_suffix(".md").name
    write_report(
        report_path    = report_path,
        html_name      = html_path.name,
        usage          = usage,
        generated_at   = generated_at,
        diff_lines     = diff_lines,
        total_sentences= len(sentences),
        null_count     = null_count,
    )

    # 组装结果
    results = []
    for i, sent in enumerate(sentences):
        results.append({
            "book":       book_id,
            "paragraph":  sent["paragraph"],
            "word_begin": sent["word_begin"],
            "word_end":   sent["word_end"],
            "pali":       sent["text"],
            "chinese":    chinese_groups[i] if i < len(chinese_groups) else None,
        })

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 入口
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM 对齐汉巴文本，输出 JSONL + 报告")
    p.add_argument("--corpus", required=True, help="语料库名称，如 milinda")
    p.add_argument("--start",  type=int, default=None, help="起始文件编号（如 1 对应 001.html）")
    p.add_argument("--end",    type=int, default=None, help="结束文件编号（如 36 对应 036.html）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    html_dir  = ROOT / "html"  / args.corpus
    meta_path = html_dir / "meta.toml"
    map_path  = html_dir / "para_map.jsonl"
    out_dir   = ROOT / "jsonl" / args.corpus
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 检查前置文件 ───────────────────────────────────────────────────────────
    if not meta_path.exists():
        log.error("找不到 %s，请先运行 01_download.py", meta_path)
        sys.exit(1)

    if not map_path.exists():
        log.error("找不到 %s，请先运行 03_scan_paragraphs.py", map_path)
        sys.exit(1)

    meta    = toml.load(meta_path)
    book_id = meta["book_id"]

    # ── 加载对照表 ─────────────────────────────────────────────────────────────
    para_map = load_para_map(map_path)
    log.info("para_map.jsonl 已加载，共 %d 条", len(para_map))

    # ── 收集要处理的 HTML 文件 ─────────────────────────────────────────────────
    html_files = sorted(html_dir.glob("*.html"))
    if not html_files:
        log.error("html/%s/ 下没有 .html 文件", args.corpus)
        sys.exit(1)

    if args.start is not None or args.end is not None:
        def _no(p: Path) -> int:
            try: return int(p.stem)
            except ValueError: return -1
        lo = args.start or 1
        hi = args.end   or 999999
        html_files = [p for p in html_files if lo <= _no(p) <= hi]

    log.info("待处理 %d 个文件（start=%s end=%s）", len(html_files), args.start, args.end)

    # ── 数据库连接 ─────────────────────────────────────────────────────────────
    conn = connect_db()
    log.info("数据库连接成功")

    # ── 主循环 ────────────────────────────────────────────────────────────────
    total_written = 0
    first_para    = None
    last_para     = None

    for html_path in html_files:
        chapter_label = f"{args.corpus}/{html_path.name}"

        paragraphs = para_map.get(html_path.name)
        if not paragraphs:
            log.warning(
                "%s 不在 para_map.jsonl 中，请先运行 03_scan_paragraphs.py",
                html_path.name,
            )
            continue

        log.info(
            "══════ 处理 %s（段落 %d..%d）══════",
            chapter_label, paragraphs[0], paragraphs[-1],
        )

        try:
            results = process_chapter(
                conn, book_id, html_path, paragraphs, chapter_label,
                report_dir=out_dir,
            )
        except Exception as e:
            log.warning("%s 处理失败，跳过: %s", chapter_label, e)
            continue

        # 写入同名 jsonl
        out_path = out_dir / html_path.with_suffix(".jsonl").name
        with out_path.open("w", encoding="utf-8") as out_f:
            for record in results:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if first_para is None:
                    first_para = record["paragraph"]
                last_para = record["paragraph"]

        total_written += len(results)
        log.info("  写入 %d 条 → %s，累计 %d 条", len(results), out_path.name, total_written)

    # ── 回填 meta.toml ─────────────────────────────────────────────────────────
    if first_para is not None:
        meta["para_start"] = first_para
        meta["para_end"]   = last_para
        with meta_path.open("w", encoding="utf-8") as f:
            toml.dump(meta, f)
        log.info("meta.toml 已更新：para_start=%d  para_end=%d", first_para, last_para)

    conn.close()
    log.info("完成，共写入 %d 条", total_written)


if __name__ == "__main__":
    main()
