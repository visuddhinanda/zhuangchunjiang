"""
extract.py
==========
第一步：从 html/{corpus}/*.html 提取中文译文，
写入 chunk/{corpus}/*.txt，并将 para_map.jsonl 复制到同一目录。

运行后，可手工切分 chunk/{corpus}/*.txt，
再由 align.py 第二步读取并对齐。
"""

import csv
import json
import logging
import re
import shutil
import sys
from html.parser import HTMLParser
from pathlib import Path

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
            mouseover = (
                dict(attrs).get("onmouseover")
                or dict(attrs).get("onMouseover")
                or ""
            )

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
                    logger.debug(
                        "glossary 无条目 id=%s，保留原文：%s", self._note_id, inner
                    )
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
# 入口
# ═════════════════════════════════════════════════════════════════════════════

def launch(corpus: str, start: int | None, end: int | None) -> None:
    """提取译文入口（无需 db）。"""
    logger.info("extract: corpus=%s start=%s end=%s", corpus, start, end)

    html_dir  = ROOT / "html"  / corpus
    meta_path = html_dir / "meta.json"
    map_path  = html_dir / "para_map.jsonl"
    chunk_dir = ROOT / "chunk" / corpus
    chunk_dir.mkdir(parents=True, exist_ok=True)

    if not meta_path.exists():
        logger.error("找不到 %s，请先运行 download", meta_path)
        sys.exit(1)

    if not map_path.exists():
        logger.error("找不到 %s，请先运行 scan_paragraphs", map_path)
        sys.exit(1)

    # ── 转换并写入 para_map.jsonl ──────────────────────────────────────────
    dest_map = chunk_dir / "para_map.jsonl"
    with map_path.open(encoding="utf-8") as src, \
         dest_map.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if line:
                rec = json.loads(line)
                txt_name = Path(rec.pop("html")).with_suffix(".txt").name
                rec = {"text": txt_name, **rec}
                dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("para_map.jsonl 已转换写入 → %s", dest_map)

    # ── 加载术语表 ─────────────────────────────────────────────────────────
    glossary = load_glossary(ROOT)

    # ── 筛选文件 ───────────────────────────────────────────────────────────
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

    logger.info("待提取 %d 个文件（start=%s end=%s）", len(html_files), start, end)

    # ── 逐文件提取 ─────────────────────────────────────────────────────────
    total_written = 0
    for html_path in html_files:
        chapter_label = f"{corpus}/{html_path.name}"
        logger.info("══════ 提取 %s ══════", chapter_label)

        try:
            html        = html_path.read_text(encoding="utf-8")
            local_notes = extract_local_notes(html)
            chinese     = extract_text(
                html, "center",
                glossary=glossary,
                local_notes=local_notes,
            )
        except Exception:
            logger.exception("%s 提取失败，跳过", chapter_label)
            continue

        if not chinese.strip():
            logger.warning("%s: #center 为空，跳过", chapter_label)
            continue

        out_path = chunk_dir / html_path.with_suffix(".txt").name
        out_path.write_text(chinese, encoding="utf-8")
        total_written += 1
        logger.info("  已写入 → %s（%d 字符）", out_path.name, len(chinese))

    logger.info("完成，共写入 %d 个 txt 文件", total_written)