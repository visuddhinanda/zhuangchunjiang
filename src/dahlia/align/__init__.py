"""
align.py
========
从 html/{corpus}/para_map.jsonl 读取 HTML↔段落对照表，
查询对应的 pali_sentences，使用 LLM 将中文译文对齐至各句，
结果写入 jsonl/{corpus}/{html_stem}.jsonl。
同时生成 jsonl/{corpus}/{html_stem}.md 对齐质量报告。
"""

import csv
import difflib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

from sqlalchemy import text

from .llm_align import LlmUsage, llm_split

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ═════════════════════════════════════════════════════════════════════════════
# 载入术语表
# ═════════════════════════════════════════════════════════════════════════════

def load_glossary(root: Path) -> dict[int, str]:
    path = root / "jsonl" / "glossary.csv"
    glossary: dict[int, str] = {}
    if not path.exists():
        logger.warning("glossary.csv 不存在：%s", path)
        return glossary
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gid = int(row["id"])
            except (KeyError, ValueError):
                continue
            pali = (row.get("pali_word") or "").strip()
            if pali:
                glossary[gid] = pali
    logger.info("glossary 已加载，共 %d 条", len(glossary))
    return glossary


# ═════════════════════════════════════════════════════════════════════════════
# 提取 local 注释表
# ═════════════════════════════════════════════════════════════════════════════

_LOCAL_SPAN_RE = re.compile(
    r'<span\s+id="note(\d+)">(.*?)</span>',
    re.IGNORECASE | re.DOTALL,
)

def extract_local_notes(html: str) -> dict[int, str]:
    notes: dict[int, str] = {}
    for m in _LOCAL_SPAN_RE.finditer(html):
        nid  = int(m.group(1))
        text = m.group(2).strip()
        notes[nid] = text
    return notes


# ═════════════════════════════════════════════════════════════════════════════
# HTML 解析
# ═════════════════════════════════════════════════════════════════════════════

_NOTE_RE  = re.compile(r"note\(this\s*,\s*(\d+)\s*\)")
_LOCAL_RE = re.compile(r"local\(this\s*,\s*(\d+)\s*\)")


class DivExtractor(HTMLParser):

    def __init__(
        self,
        target_id:   str,
        glossary:    dict[int, str] | None = None,
        local_notes: dict[int, str] | None = None,
    ):
        super().__init__()
        self.target_id   = target_id
        self.glossary    = glossary    or {}
        self.local_notes = local_notes or {}
        self.depth       = 0
        self.in_target   = False
        self.parts: list[str] = []

        self._note_id:   int | None = None
        self._in_note:   bool = False
        self._note_text: list[str] = []

        self._local_id:   int | None = None
        self._in_local:   bool = False
        self._local_text: list[str] = []

        self._note_count  = 0
        self._local_count = 0

    def handle_starttag(self, tag: str, attrs):
        if not self.in_target:
            if dict(attrs).get("id") == self.target_id:
                self.in_target = True
                self.depth = 1
            return

        self.depth += 1

        if tag == "br":
            self.parts.append("\n")
            return

        if tag == "a" and not self._in_note and not self._in_local:
            mouseover = dict(attrs).get("onmouseover") or dict(attrs).get("onMouseover") or ""

            m = _NOTE_RE.search(mouseover)
            if m:
                self._note_id   = int(m.group(1))
                self._in_note   = True
                self._note_text = []
                return

            m = _LOCAL_RE.search(mouseover)
            if m:
                self._local_id   = int(m.group(1))
                self._in_local   = True
                self._local_text = []

    def handle_endtag(self, tag: str):
        if not self.in_target:
            return

        if tag == "a":
            if self._in_note:
                inner = "".join(self._note_text)
                pali  = self.glossary.get(self._note_id)  # type: ignore[arg-type]
                if pali:
                    self.parts.append(f"[[{pali}#{inner}]]")
                    self._note_count += 1
                else:
                    self.parts.append(inner)
                    logger.debug("glossary 无条目 id=%s，保留原文：%s", self._note_id, inner)
                self._in_note   = False
                self._note_id   = None
                self._note_text = []

            elif self._in_local:
                trigger   = "".join(self._local_text)
                note_text = self.local_notes.get(self._local_id)  # type: ignore[arg-type]
                if note_text is not None:
                    self.parts.append(
                        f"{{{{note|trigger={trigger}|text={note_text}}}}}"
                    )
                    self._local_count += 1
                else:
                    self.parts.append(trigger)
                    logger.warning(
                        "local 注释未找到 id=%s，触发文字：%s", self._local_id, trigger
                    )
                self._in_local   = False
                self._local_id   = None
                self._local_text = []

        self.depth -= 1
        if self.depth == 0:
            self.in_target = False

    def handle_data(self, data: str):
        if not self.in_target:
            return
        if self._in_note:
            self._note_text.append(data)
        elif self._in_local:
            self._local_text.append(data)
        else:
            self.parts.append(data)

    @property
    def text(self) -> str:
        return "".join(self.parts)


def extract_text(
    html:        str,
    div_id:      str,
    glossary:    dict[int, str] | None = None,
    local_notes: dict[int, str] | None = None,
) -> str:
    p = DivExtractor(div_id, glossary=glossary, local_notes=local_notes)
    p.feed(html)
    logger.info("  注释替换：note %d 处，local %d 处", p._note_count, p._local_count)
    return p.text


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

        # 新增：位置信息
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

        # 新增：上下文
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

        # 首个不同字符
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
    html_name:       str,
    usage:           LlmUsage,
    generated_at:    str,
    diff_lines:      list[str],
    total_sentences: int,
    null_count:      int,
) -> None:
    lines: list[str] = []
    lines.append(f"# 对齐报告：{html_name}\n")

    lines.append("## 基本信息\n")
    lines.append("| 项目 | 值 |")
    lines.append("|------|----|")
    lines.append(f"| HTML 文件 | `{html_name}` |")
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
    html_path:     Path,
    paragraphs:    list[int],
    chapter_label: str,
    report_dir:    Path,
    glossary:      dict[int, str] | None = None,
) -> list[dict]:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    html          = html_path.read_text(encoding="utf-8")
    local_notes   = extract_local_notes(html)
    chinese_clean = extract_text(html, "center", glossary=glossary, local_notes=local_notes)

    if not chinese_clean.strip():
        raise ValueError(f"{chapter_label}: #center 为空")

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
    report_path = report_dir / html_path.with_suffix(".md").name
    write_report(
        report_path     = report_path,
        html_name       = html_path.name,
        usage           = usage,
        generated_at    = generated_at,
        diff_lines      = diff_lines,
        total_sentences = len(sentences),
        null_count      = null_count,
    )

    results = []
    for i, sent in enumerate(sentences):
        results.append({
            "id":      f"{book_id}-{sent["paragraph"]}-{sent["word_begin"]}-{sent["word_end"]}",
            "original":      sent["text"],
            "content":   chinese_groups[i] if i < len(chinese_groups) else None,
        })

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 入口
# ═════════════════════════════════════════════════════════════════════════════

def launch(db, corpus: str, start: int | None, end: int | None) -> None:
    logger.info("align: corpus=%s start=%s end=%s", corpus, start, end)

    html_dir  = ROOT / "html"  / corpus
    meta_path = html_dir / "meta.json"
    map_path  = html_dir / "para_map.jsonl"
    out_dir   = ROOT / "jsonl" / corpus
    out_dir.mkdir(parents=True, exist_ok=True)

    if not meta_path.exists():
        logger.error("找不到 %s，请先运行 download", meta_path)
        sys.exit(1)

    if not map_path.exists():
        logger.error("找不到 %s，请先运行 scan_paragraphs", map_path)
        sys.exit(1)

    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    book_id = meta["book_id"]

    para_map = load_para_map(map_path)
    logger.info("para_map.jsonl 已加载，共 %d 条", len(para_map))

    html_files = sorted(html_dir.glob("*.html"))
    if not html_files:
        logger.error("html/%s/ 下没有 .html 文件", corpus)
        sys.exit(1)

    if start is not None or end is not None:
        def _no(p: Path) -> int:
            try:
                return int(p.stem)
            except ValueError:
                return -1
        lo = start or 1
        hi = end   or 999999
        html_files = [p for p in html_files if lo <= _no(p) <= hi]

    logger.info("待处理 %d 个文件（start=%s end=%s）", len(html_files), start, end)

    glossary = load_glossary(ROOT)

    total_written = 0
    first_para    = None
    last_para     = None

    with db.connect() as conn:
        for html_path in html_files:
            chapter_label = f"{corpus}/{html_path.name}"

            paragraphs = para_map.get(html_path.name)
            if not paragraphs:
                logger.warning(
                    "%s 不在 para_map.jsonl 中，请先运行 scan_paragraphs",
                    html_path.name,
                )
                continue

            logger.info(
                "══════ 处理 %s（段落 %d..%d）══════",
                chapter_label, paragraphs[0], paragraphs[-1],
            )

            try:
                results = process_chapter(
                    conn, book_id, html_path, paragraphs, chapter_label,
                    report_dir=out_dir,
                    glossary=glossary,
                )
            except Exception as e:
                logger.exception("%s 处理失败，跳过", chapter_label)
                continue

            out_path = out_dir / html_path.with_suffix(".jsonl").name
            with out_path.open("w", encoding="utf-8") as out_f:
                for record in results:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    id = [int(x) for x in record["id"].split("-")]
                    if first_para is None:
                        first_para = id[1]
                    last_para = id[1]

            total_written += len(results)
            logger.info(
                "  写入 %d 条 → %s，累计 %d 条",
                len(results), out_path.name, total_written,
            )

    # ── 回填 meta.json ─────────────────────────────────────────────────────────
    if first_para is not None:
        meta["para_start"] = first_para
        meta["para_end"]   = last_para
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info("meta.json 已更新：para_start=%d  para_end=%d", first_para, last_para)

    logger.info("完成，共写入 %d 条", total_written)