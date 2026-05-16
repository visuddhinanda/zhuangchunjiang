"""
align.py
========
第二步：从 chunk/{corpus}/*.txt 读取中文译文（可手工切分），
查询对应的 pali_sentences，使用 LLM 将中文译文对齐至各句，
结果写入 jsonl/{corpus}/{stem}.jsonl。
同时生成 jsonl/{corpus}/{stem}.md 对齐质量报告。

第一步（提取译文）见 extract.py。
"""

import difflib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text


from .llm_align import LlmUsage, llm_split

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ═════════════════════════════════════════════════════════════════════════════
# 对照表加载
# ═════════════════════════════════════════════════════════════════════════════

def load_para_map(map_path: Path) -> dict[str, list[int]]:
    if not map_path.exists():
        return {}
    result: dict[str, list[int]] = {}
    with map_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                result[rec["text"]] = list(range(rec["para_start"], rec["para_end"] + 1))
    return result

# ═════════════════════════════════════════════════════════════════════════════
# 文本正规化
# ═════════════════════════════════════════════════════════════════════════════

def normalize_for_diff(t: str) -> str:
    return re.sub(r"[\s\u3000]+", "", t)


# ═════════════════════════════════════════════════════════════════════════════
# 数据库
# ═════════════════════════════════════════════════════════════════════════════

def load_pali_sentences(conn, book_id: int, paragraphs: list[int]) -> list[dict]:
    result = conn.execute(
        text("""
            SELECT id, paragraph, word_begin, word_end, text
            FROM pali_sentences
            WHERE book = :book AND paragraph = ANY(:paras)
            ORDER BY paragraph, word_begin
        """),
        {"book": book_id, "paras": paragraphs},
    )
    return [dict(r._mapping) for r in result.fetchall()]


# ═════════════════════════════════════════════════════════════════════════════
# Diff 比对
# ═════════════════════════════════════════════════════════════════════════════

def compute_diff(original: str, aligned: str) -> list[str]:
    orig_norm    = normalize_for_diff(original)
    aligned_norm = normalize_for_diff(aligned)

    if orig_norm == aligned_norm:
        return []

    matcher = difflib.SequenceMatcher(
        None,
        orig_norm,
        aligned_norm,
        autojunk=False,
    )

    diffs: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        orig_chunk    = orig_norm[i1:i2]
        aligned_chunk = aligned_norm[j1:j2]

        diffs.append(
            f"@@ 原文[{i1}:{i2}] 对齐[{j1}:{j2}] tag={tag}"
        )

        if tag == "replace":
            diffs.append(f"- 原文: `{orig_chunk}`")
            diffs.append(f"+ 对齐: `{aligned_chunk}`")
        elif tag == "delete":
            diffs.append(f"- 原文多出: `{orig_chunk}`")
        elif tag == "insert":
            diffs.append(f"+ 对齐多出: `{aligned_chunk}`")

        context_before_orig = orig_norm[max(0, i1 - 20):i1]
        context_after_orig  = orig_norm[i2:i2 + 20]
        diffs.append(
            f"  原文上下文: ...{context_before_orig}"
            f"[{orig_chunk}]"
            f"{context_after_orig}..."
        )

        context_before_aligned = aligned_norm[max(0, j1 - 20):j1]
        context_after_aligned  = aligned_norm[j2:j2 + 20]
        diffs.append(
            f"  对齐上下文: ...{context_before_aligned}"
            f"[{aligned_chunk}]"
            f"{context_after_aligned}..."
        )

        if orig_chunk and aligned_chunk:
            for k, (a, b) in enumerate(zip(orig_chunk, aligned_chunk)):
                if a != b:
                    diffs.append(
                        f"  ℹ️ 首个不同字符位置 {k}: "
                        f"原文 U+{ord(a):04X}({a!r}) "
                        f"vs "
                        f"对齐 U+{ord(b):04X}({b!r})"
                    )
                    break

    return diffs


# ═════════════════════════════════════════════════════════════════════════════
# Markdown 报告
# ═════════════════════════════════════════════════════════════════════════════

def write_report(
    report_path:     Path,
    source_name:     str,
    usage:           LlmUsage,
    generated_at:    str,
    diff_lines:      list[str],
    total_sentences: int,
    null_count:      int,
) -> None:
    lines: list[str] = []
    lines.append(f"# 对齐报告：{source_name}\n")

    lines.append("## 基本信息\n")
    lines.append("| 项目 | 值 |")
    lines.append("|------|----|")
    lines.append(f"| 源文件 | `{source_name}` |")
    lines.append(f"| 生成时间 | {generated_at} |")
    lines.append(f"| 模型 | `{usage.model}` |")
    lines.append(f"| 句子总数 | {total_sentences} |")
    lines.append(f"| 未对齐（null）| {null_count} |")
    lines.append("")

    lines.append("## Token 用量\n")
    lines.append("| 类型 | 数量 |")
    lines.append("|------|------|")
    lines.append(f"| Prompt tokens | {usage.prompt_tokens} |")
    lines.append(f"| Completion tokens | {usage.completion_tokens} |")
    lines.append(f"| Total tokens | {usage.total_tokens} |")
    if usage.extra:
        for k, v in usage.extra.items():
            lines.append(f"| {k} | {v} |")
    lines.append("")

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
    logger.info("  报告已写入 → %s", report_path.name)


# ═════════════════════════════════════════════════════════════════════════════
# 核心处理
# ═════════════════════════════════════════════════════════════════════════════

def process_chapter(
    conn,
    book_id:       int,
    txt_path:      Path,
    paragraphs:    list[int],
    chapter_label: str,
    report_dir:    Path,
) -> list[dict]:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    chinese_clean = txt_path.read_text(encoding="utf-8")
    if not chinese_clean.strip():
        raise ValueError(f"{chapter_label}: txt 文件为空")

    sentences = load_pali_sentences(conn, book_id, paragraphs)
    if not sentences:
        raise ValueError(f"{chapter_label}: pali_sentences 为空，段落={paragraphs}")

    logger.info("  pali_sentences 共 %d 条", len(sentences))

    chinese_groups, usage = llm_split(chinese_clean, sentences)

    aligned_text = "".join(g for g in chinese_groups if g is not None)
    diff_lines   = compute_diff(chinese_clean, aligned_text)

    if diff_lines:
        logger.warning("  文本比对：发现 %d 处差异", len(diff_lines))
    else:
        logger.info("  文本比对：无差异 ✅")

    null_count  = sum(1 for g in chinese_groups if g is None)
    report_path = report_dir / txt_path.with_suffix(".md").name
    write_report(
        report_path     = report_path,
        source_name     = txt_path.name,
        usage           = usage,
        generated_at    = generated_at,
        diff_lines      = diff_lines,
        total_sentences = len(sentences),
        null_count      = null_count,
    )

    results = []
    for i, sent in enumerate(sentences):
        results.append({
            "id":       f"{book_id}-{sent['paragraph']}-{sent['word_begin']}-{sent['word_end']}",
            "original": sent["text"],
            "content":  chinese_groups[i] if i < len(chinese_groups) else None,
        })

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 入口
# ═════════════════════════════════════════════════════════════════════════════

def launch(db, corpus: str, start: int | None, end: int | None) -> None:
    logger.info("align: corpus=%s start=%s end=%s", corpus, start, end)

    html_dir  = ROOT / "html"  / corpus
    meta_path = html_dir / "meta.json"
    chunk_dir = ROOT / "chunk" / corpus
    map_path  = chunk_dir / "para_map.jsonl"
    out_dir   = ROOT / "jsonl" / corpus
    out_dir.mkdir(parents=True, exist_ok=True)

    if not meta_path.exists():
        logger.error("找不到 %s，请先运行 download", meta_path)
        sys.exit(1)

    if not chunk_dir.exists():
        logger.error("找不到 chunk 目录 %s，请先运行 extract", chunk_dir)
        sys.exit(1)

    if not map_path.exists():
        logger.error("找不到 %s，请先运行 extract", map_path)
        sys.exit(1)

    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    book_id = meta["book_id"]

    para_map = load_para_map(map_path)
    logger.info("para_map.jsonl 已加载，共 %d 条", len(para_map))

    txt_files = sorted(chunk_dir.glob("*.txt"))
    if not txt_files:
        logger.error("chunk/%s/ 下没有 .txt 文件", corpus)
        sys.exit(1)

    # 按 para_map 行号筛选
    all_entries = list(para_map.items())  # [(txt_name, paragraphs), ...]
    lo = (start or 1) - 1          # 转为 0-based
    hi = (end   or len(all_entries)) - 1
    selected = all_entries[lo:hi + 1]

    logger.info("待处理 %d 个文件（start=%s end=%s）", len(selected), start, end)

    total_written = 0
    first_para    = None
    last_para     = None

    with db.connect() as conn:
        for txt_name, paragraphs in selected:
            txt_path      = chunk_dir / txt_name
            chapter_label = f"{corpus}/{txt_name}"

            if not txt_path.exists():
                logger.warning("%s 文件不存在，跳过", txt_path)
                continue

            logger.info(
                "══════ 处理 %s（段落 %d..%d）══════",
                chapter_label, paragraphs[0], paragraphs[-1],
            )

            try:
                results = process_chapter(
                    conn, book_id, txt_path, paragraphs, chapter_label,
                    report_dir=out_dir,
                )
            except Exception:
                logger.exception("%s 处理失败，跳过", chapter_label)
                continue

            out_path = out_dir / txt_path.with_suffix(".jsonl").name
            with out_path.open("w", encoding="utf-8") as out_f:
                for record in results:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    id_parts = [int(x) for x in record["id"].split("-")]
                    if first_para is None:
                        first_para = id_parts[1]
                    last_para = id_parts[1]

            total_written += len(results)
            logger.info(
                "  写入 %d 条 → %s，累计 %d 条",
                len(results), out_path.name, total_written,
            )

    # ── 回填 meta.json ─────────────────────────────────────────────────────
    if first_para is not None:
        meta["para_start"] = first_para
        meta["para_end"]   = last_para
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info("meta.json 已更新：para_start=%d  para_end=%d", first_para, last_para)

    logger.info("完成，共写入 %d 条", total_written)