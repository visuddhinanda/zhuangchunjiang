"""
align.py
========
第二步：从 chunk/{corpus}/*.txt 读取中文译文（可手工切分），
查询对应的 pali_sentences，使用 LLM 将中文译文对齐至各句，
结果写入 jsonl/{corpus}/{stem}.jsonl。
同时生成 jsonl/{corpus}/{stem}.md 对齐质量报告，
并维护 jsonl/{corpus}/{corpus}_summary.csv 总表。

第一步（提取译文）见 extract.py。
"""

import csv
import difflib
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text

from .llm_align import LlmUsage, llm_split

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent.parent

SUMMARY_FIELDS = [
    "序号", "文件名", "句子总数", "中文字符数",
    "模型", "Prompt tokens", "Completion tokens",
    "原始差异", "修正后剩余差异", "生成时间",
]


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

@dataclass
class DiffHunk:
    """一处差异的结构化表示。"""
    tag:           str
    orig_start:    int
    orig_end:      int
    aligned_start: int
    aligned_end:   int
    orig_chunk:    str
    aligned_chunk: str


def compute_diff(original: str, aligned: str) -> tuple[list[DiffHunk], list[str]]:
    """
    返回 (hunks, log_lines)。
    hunks     — 结构化差异列表，供自动修正使用。
    log_lines — 人类可读的 diff 行，供报告使用。
    """
    orig_norm    = normalize_for_diff(original)
    aligned_norm = normalize_for_diff(aligned)

    if orig_norm == aligned_norm:
        return [], []

    matcher = difflib.SequenceMatcher(
        None,
        orig_norm,
        aligned_norm,
        autojunk=False,
    )

    hunks:     list[DiffHunk] = []
    log_lines: list[str]      = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        orig_chunk    = orig_norm[i1:i2]
        aligned_chunk = aligned_norm[j1:j2]

        hunks.append(DiffHunk(
            tag           = tag,
            orig_start    = i1,
            orig_end      = i2,
            aligned_start = j1,
            aligned_end   = j2,
            orig_chunk    = orig_chunk,
            aligned_chunk = aligned_chunk,
        ))

        log_lines.append(f"@@ 原文[{i1}:{i2}] 对齐[{j1}:{j2}] tag={tag}")

        if tag == "replace":
            log_lines.append(f"- 原文: `{orig_chunk}`")
            log_lines.append(f"+ 对齐: `{aligned_chunk}`")
        elif tag == "delete":
            log_lines.append(f"- 原文多出: `{orig_chunk}`")
        elif tag == "insert":
            log_lines.append(f"+ 对齐多出: `{aligned_chunk}`")

        context_before_orig = orig_norm[max(0, i1 - 20):i1]
        context_after_orig  = orig_norm[i2:i2 + 20]
        log_lines.append(
            f"  原文上下文: ...{context_before_orig}"
            f"[{orig_chunk}]"
            f"{context_after_orig}..."
        )

        context_before_aligned = aligned_norm[max(0, j1 - 20):j1]
        context_after_aligned  = aligned_norm[j2:j2 + 20]
        log_lines.append(
            f"  对齐上下文: ...{context_before_aligned}"
            f"[{aligned_chunk}]"
            f"{context_after_aligned}..."
        )

        if orig_chunk and aligned_chunk:
            for k, (a, b) in enumerate(zip(orig_chunk, aligned_chunk)):
                if a != b:
                    log_lines.append(
                        f"  ℹ️ 首个不同字符位置 {k}: "
                        f"原文 U+{ord(a):04X}({a!r}) "
                        f"vs "
                        f"对齐 U+{ord(b):04X}({b!r})"
                    )
                    break

    return hunks, log_lines


# ═════════════════════════════════════════════════════════════════════════════
# 自动修正
# ═════════════════════════════════════════════════════════════════════════════

def apply_auto_corrections(
    results: list[dict],
    hunks:   list[DiffHunk],
) -> tuple[list[dict], list[str], list[str]]:
    """
    对 results 中的 content 做定点修正。

    处理 replace 且原文/对齐片段均小于 4 个字符的差异：
    - 在归一化对齐文中按偏移量定位到对应的 content 条目；
    - 在原始 content 字符串中找到该归一化位置对应的真实位置，执行替换。

    返回 (corrected_results, fixed_log, skipped_log)。
    """
    def is_fixable(h: DiffHunk) -> bool:
        return h.tag == "replace" and len(h.orig_chunk) < 4 and len(h.aligned_chunk) < 4

    fixable   = [h for h in hunks if is_fixable(h)]
    unfixable = [h for h in hunks if not is_fixable(h)]

    fixed_log   = [f"自动修正 {len(fixable)} 处差异："]
    skipped_log = [f"跳过 {len(unfixable)} 处复杂差异（需人工检查）："]

    for h in unfixable:
        skipped_log.append(
            f"  tag={h.tag} 对齐[{h.aligned_start}:{h.aligned_end}] "
            f"原文片段=`{h.orig_chunk}` 对齐片段=`{h.aligned_chunk}`"
        )

    if not fixable:
        return results, fixed_log, skipped_log

    fixable_sorted = sorted(fixable, key=lambda h: h.aligned_start)

    norm_cursor = 0
    fix_idx     = 0
    contents    = [rec.get("content") or "" for rec in results]

    for rec_idx, content in enumerate(contents):
        if fix_idx >= len(fixable_sorted):
            break
        if not content:
            continue

        content_norm_len = len(normalize_for_diff(content))
        norm_end         = norm_cursor + content_norm_len

        while fix_idx < len(fixable_sorted):
            h = fixable_sorted[fix_idx]
            if h.aligned_start >= norm_end:
                break

            offset_in_norm = h.aligned_start - norm_cursor
            chunk_len      = len(h.aligned_chunk)

            # 收集本条 content 中所有非空白字符的真实位置
            real_positions: list[int] = [
                i for i, ch in enumerate(content)
                if not re.match(r"[\s\u3000]", ch)
            ]

            target_positions = real_positions[offset_in_norm: offset_in_norm + chunk_len]

            if len(target_positions) != chunk_len:
                fixed_log.append(
                    f"  record[{rec_idx}]: 归一化偏移越界，跳过"
                )
                fix_idx += 1
                continue

            # 验证每个位置的字符与 aligned_chunk 一致
            mismatch = False
            for k, real_pos in enumerate(target_positions):
                if content[real_pos] != h.aligned_chunk[k]:
                    fixed_log.append(
                        f"  record[{rec_idx}]: 位置 {real_pos} 预期 `{h.aligned_chunk[k]}`"
                        f" 实际 `{content[real_pos]}`，整处跳过"
                    )
                    mismatch = True
                    break

            if not mismatch:
                # 从后往前删除对齐片段占据的位置，再在起始处插入原文片段
                content_list = list(content)
                for real_pos in reversed(target_positions):
                    del content_list[real_pos]
                insert_at = target_positions[0]
                content_list[insert_at:insert_at] = list(h.orig_chunk)
                content           = "".join(content_list)
                contents[rec_idx] = content
                fixed_log.append(
                    f"  record[{rec_idx}] 位置 {target_positions}: "
                    f"`{h.aligned_chunk}` → `{h.orig_chunk}`"
                )

            fix_idx += 1

        norm_cursor = norm_end

    for rec_idx, content in enumerate(contents):
        if results[rec_idx].get("content") is not None:
            results[rec_idx]["content"] = content

    return results, fixed_log, skipped_log


# ═════════════════════════════════════════════════════════════════════════════
# Markdown 报告
# ═════════════════════════════════════════════════════════════════════════════

def write_report(
    report_path:      Path,
    source_name:      str,
    usage:            LlmUsage,
    generated_at:     str,
    hunks:            list[DiffHunk],
    log_lines:        list[str],
    fixed_log:        list[str],
    skipped_log:      list[str],
    log_lines_fixed:  list[str],
    total_sentences:  int,
    null_count:       int,
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

    lines.append("## 原始差异\n")
    if not log_lines:
        lines.append("✅ 无差异：对齐后拼接文本与原文完全一致（忽略空白）。")
    else:
        lines.append(f"发现 {len(hunks)} 处差异（已忽略空白）：\n")
        lines.append("```diff")
        lines.extend(log_lines)
        lines.append("```")
    lines.append("")

    lines.append("## 自动修正\n")
    lines.extend(fixed_log)
    lines.append("")
    if skipped_log[1:]:
        lines.extend(skipped_log)
        lines.append("")

    lines.append("## 修正后剩余差异\n")
    if not log_lines_fixed:
        lines.append("✅ 无剩余差异。")
    else:
        remaining = sum(1 for l in log_lines_fixed if l.startswith("@@"))
        lines.append(f"剩余 {remaining} 处差异（需人工检查）：\n")
        lines.append("```diff")
        lines.extend(log_lines_fixed)
        lines.append("```")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("  报告已写入 → %s", report_path.name)


# ═════════════════════════════════════════════════════════════════════════════
# 总表 CSV
# ═════════════════════════════════════════════════════════════════════════════

def load_summary_csv(csv_path: Path) -> dict[str, dict]:
    """读取已有 CSV，以文件名为 key 返回字典。"""
    row_data: dict[str, dict] = {}
    if not csv_path.exists():
        return row_data
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("文件名", "")
            if name:
                row_data[name] = dict(row)
    return row_data


def write_summary_csv(
    csv_path:    Path,
    all_entries: list[tuple[str, list[int]]],
    row_data:    dict[str, dict],
) -> None:
    """按 para_map 行序整体重写 CSV，未处理的行保留历史数据或仅填序号和文件名。"""
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for idx, (txt_name, _) in enumerate(all_entries, start=1):
            if txt_name in row_data:
                row = dict(row_data[txt_name])
                row["序号"]  = idx
                row["文件名"] = txt_name
                writer.writerow(row)
            else:
                writer.writerow({"序号": idx, "文件名": txt_name})


# ═════════════════════════════════════════════════════════════════════════════
# 核心处理
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ChapterResult:
    records:           list[dict]
    total_sentences:   int
    char_count:        int
    model:             str
    prompt_tokens:     int
    completion_tokens: int
    orig_diff_count:   int
    fixed_diff_count:  int
    generated_at:      str


def process_chapter(
    conn,
    book_id:       int,
    txt_path:      Path,
    paragraphs:    list[int],
    chapter_label: str,
    report_dir:    Path,
) -> ChapterResult:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    chinese_clean = txt_path.read_text(encoding="utf-8")
    if not chinese_clean.strip():
        raise ValueError(f"{chapter_label}: txt 文件为空")

    sentences = load_pali_sentences(conn, book_id, paragraphs)
    if not sentences:
        raise ValueError(f"{chapter_label}: pali_sentences 为空，段落={paragraphs}")

    logger.info("  pali_sentences 共 %d 条", len(sentences))

    chinese_groups, usage = llm_split(chinese_clean, sentences)

    results = []
    for i, sent in enumerate(sentences):
        results.append({
            "id":       f"{book_id}-{sent['paragraph']}-{sent['word_begin']}-{sent['word_end']}",
            "original": sent["text"],
            "content":  chinese_groups[i] if i < len(chinese_groups) else None,
        })

    # ── 初次 diff ─────────────────────────────────────────────────────────
    aligned_text     = "".join(r["content"] for r in results if r["content"])
    hunks, log_lines = compute_diff(chinese_clean, aligned_text)

    if hunks:
        logger.warning("  文本比对：发现 %d 处差异", len(hunks))
    else:
        logger.info("  文本比对：无差异 ✅")

    # ── 自动修正 ──────────────────────────────────────────────────────────
    results, fixed_log, skipped_log = apply_auto_corrections(results, hunks)

    # ── 修正后再次 diff ───────────────────────────────────────────────────
    aligned_text_fixed = "".join(r["content"] for r in results if r["content"])
    _, log_lines_fixed = compute_diff(chinese_clean, aligned_text_fixed)
    fixed_diff_count   = sum(1 for l in log_lines_fixed if l.startswith("@@"))

    if log_lines_fixed:
        logger.warning("  修正后剩余 %d 处差异", fixed_diff_count)
    else:
        logger.info("  修正后：无剩余差异 ✅")

    null_count  = sum(1 for r in results if r["content"] is None)
    report_path = report_dir / txt_path.with_suffix(".md").name
    write_report(
        report_path     = report_path,
        source_name     = txt_path.name,
        usage           = usage,
        generated_at    = generated_at,
        hunks           = hunks,
        log_lines       = log_lines,
        fixed_log       = fixed_log,
        skipped_log     = skipped_log,
        log_lines_fixed = log_lines_fixed,
        total_sentences = len(sentences),
        null_count      = null_count,
    )

    return ChapterResult(
        records           = results,
        total_sentences   = len(sentences),
        char_count        = len(chinese_clean),
        model             = usage.model,
        prompt_tokens     = usage.prompt_tokens,
        completion_tokens = usage.completion_tokens,
        orig_diff_count   = len(hunks),
        fixed_diff_count  = fixed_diff_count,
        generated_at      = generated_at,
    )


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

    all_entries = list(para_map.items())
    if not all_entries:
        logger.error("chunk/%s/para_map.jsonl 为空", corpus)
        sys.exit(1)

    lo       = (start or 1) - 1
    hi       = (end   or len(all_entries)) - 1
    selected = all_entries[lo:hi + 1]

    logger.info("待处理 %d 个文件（start=%s end=%s）", len(selected), start, end)

    # ── 读取已有总表 ───────────────────────────────────────────────────────
    csv_path = out_dir / f"{corpus}_summary.csv"
    row_data = load_summary_csv(csv_path)

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
                cr = process_chapter(
                    conn, book_id, txt_path, paragraphs, chapter_label,
                    report_dir=out_dir,
                )
            except Exception:
                logger.exception("%s 处理失败，跳过", chapter_label)
                continue

            # ── 写入 JSONL ────────────────────────────────────────────────
            out_path = out_dir / txt_path.with_suffix(".jsonl").name
            with out_path.open("w", encoding="utf-8") as out_f:
                for record in cr.records:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    id_parts = [int(x) for x in record["id"].split("-")]
                    if first_para is None:
                        first_para = id_parts[1]
                    last_para = id_parts[1]

            total_written += len(cr.records)
            logger.info(
                "  写入 %d 条 → %s，累计 %d 条",
                len(cr.records), out_path.name, total_written,
            )

            # ── 更新总表 ──────────────────────────────────────────────────
            row_data[txt_name] = {
                "序号":             0,          # 由 write_summary_csv 按行序填入
                "文件名":           txt_name,
                "句子总数":         cr.total_sentences,
                "中文字符数":       cr.char_count,
                "模型":             cr.model,
                "Prompt tokens":    cr.prompt_tokens,
                "Completion tokens": cr.completion_tokens,
                "原始差异":         cr.orig_diff_count,
                "修正后剩余差异":   cr.fixed_diff_count,
                "生成时间":         cr.generated_at,
            }
            write_summary_csv(csv_path, all_entries, row_data)
            logger.info("  总表已更新 → %s", csv_path.name)

    # ── 回填 meta.json ─────────────────────────────────────────────────────
    if first_para is not None:
        meta["para_start"] = first_para
        meta["para_end"]   = last_para
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info("meta.json 已更新：para_start=%d  para_end=%d", first_para, last_para)

    logger.info("完成，共写入 %d 条", total_written)