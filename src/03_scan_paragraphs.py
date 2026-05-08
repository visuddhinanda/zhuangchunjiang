#!/usr/bin/env python3
"""
03_scan_paragraphs.py
=====================
扫描 html/{corpus}/ 下的 HTML 文件，逐一与 WikiPali pali_texts 表做单步逼近匹配，
生成每个 HTML 文件对应的巴利文段落范围，保存至 html/{corpus}/para_map.jsonl。

para_map.jsonl 每行格式：
    {
        "html": "001.html",
        "paragraphs": [1, 2, 3, ..., 17]
    }

此文件供 02_align.py 使用，解耦"段落扫描"与"文本对齐"两个步骤。

用法：
    python src/03_scan_paragraphs.py --corpus milinda
    python src/03_scan_paragraphs.py --corpus milinda --start 1 --end 5
"""

import argparse
import json
import logging
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import psycopg2
import psycopg2.extras
import toml
from dotenv import load_dotenv

# ── 日志配置 ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


# ═════════════════════════════════════════════════════════════════════════════
# HTML 解析（只需巴利文侧）
# ═════════════════════════════════════════════════════════════════════════════

class DivExtractor(HTMLParser):
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


def extract_pali(html: str) -> str:
    p = DivExtractor("east")
    p.feed(html)
    return p.text


# ═════════════════════════════════════════════════════════════════════════════
# 文本工具
# ═════════════════════════════════════════════════════════════════════════════

def strip_whitespace(text: str) -> str:
    return re.sub(r"[\s\u3000]+", "", text)


def normalize_for_head(text: str) -> str:
    text = strip_whitespace(text)
    text = re.sub(r"[^\w]", "", text, flags=re.UNICODE)
    return text.lower()


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


def load_one_row(conn, book_id: int, paragraph: int) -> dict | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT paragraph, text FROM pali_texts "
            "WHERE book = %s AND paragraph = %s",
            (book_id, paragraph),
        )
        row = cur.fetchone()
        return dict(row) if row else None


# ═════════════════════════════════════════════════════════════════════════════
# 核心：单步逼近
# ═════════════════════════════════════════════════════════════════════════════

def match_pali_texts(
    conn,
    book_id: int,
    cursor: int,
    pali_clean: str,
    chapter_label: str,
) -> tuple[list[int], int]:
    """
    对单个 HTML 页面的巴利文做单步逼近，返回 (matched_paragraphs, new_cursor)。

    开头验证：取两者 normalize 后较短者长度做宽松比对。
    逼近：累计 < 目标 * 0.95 时直接加入；进入临界后三向比较选最优。
    """
    # ── 开头验证 ──────────────────────────────────────────────────────────────
    first_row = load_one_row(conn, book_id, cursor)
    if not first_row:
        raise RuntimeError(f"pali_texts 已耗尽，cursor={cursor}")

    agama_norm = normalize_for_head(pali_clean)
    db_norm    = normalize_for_head(first_row["text"])
    head_len   = min(30, len(agama_norm), len(db_norm))

    if agama_norm[:head_len] != db_norm[:head_len]:
        raise RuntimeError(
            f"{chapter_label} 开头验证失败！\n"
            f"  网站: [{agama_norm[:head_len]}]\n"
            f"  数据库: [{db_norm[:head_len]}]"
        )
    log.info("  开头验证通过: %s", agama_norm[:head_len])

    # ── 单步逼近 ───────────────────────────────────────────────────────────────
    agama_len  = len(strip_whitespace(pali_clean))
    accumulated = ""
    matched: list[int] = []

    while True:
        row = load_one_row(conn, book_id, cursor)
        if not row:
            log.warning("  pali_texts 已耗尽，cursor=%d", cursor)
            break

        candidate     = accumulated + strip_whitespace(row["text"])
        candidate_len = len(candidate)
        log.debug("  指针=%d 缓冲=%d 目标=%d", cursor, candidate_len, agama_len)

        if candidate_len < agama_len:
            accumulated = candidate
            matched.append(row["paragraph"])
            cursor += 1
            continue

        # 临界区：三向比较
        prev_accumulated = accumulated
        prev_matched     = matched.copy()

        accumulated = candidate
        matched.append(row["paragraph"])
        cursor += 1

        next_row = load_one_row(conn, book_id, cursor)
        if not next_row:
            break

        candidate_next = accumulated + strip_whitespace(next_row["text"])

        diff_prev = abs(len(prev_accumulated) - agama_len)
        diff_curr = abs(len(accumulated)      - agama_len)
        diff_next = abs(len(candidate_next)   - agama_len)

        if diff_prev <= diff_curr and diff_prev <= diff_next:
            accumulated = prev_accumulated
            matched     = prev_matched
            cursor     -= 1
        elif diff_next < diff_curr:
            accumulated = candidate_next
            matched.append(next_row["paragraph"])
            cursor += 1

        break

    log.info(
        "  逼近完成 | 网站=%d 累计=%d 差异=%d | p%d..p%d",
        agama_len,
        len(accumulated),
        abs(len(accumulated) - agama_len),
        matched[0]  if matched else 0,
        matched[-1] if matched else 0,
    )
    return matched, cursor


# ═════════════════════════════════════════════════════════════════════════════
# 入口
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="扫描巴利文段落，生成 HTML↔段落对照表")
    p.add_argument("--corpus", required=True, help="语料库名称，如 milinda")
    p.add_argument("--start",  type=int, default=None, help="起始文件编号")
    p.add_argument("--end",    type=int, default=None, help="结束文件编号")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    html_dir  = ROOT / "html" / args.corpus
    meta_path = html_dir / "meta.toml"
    map_path  = html_dir / "para_map.jsonl"

    if not meta_path.exists():
        log.error("找不到 %s，请先运行 01_download.py", meta_path)
        sys.exit(1)

    meta    = toml.load(meta_path)
    book_id = meta["book_id"]

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

    log.info("共 %d 个 HTML 文件待扫描", len(html_files))

    # ── 数据库 ────────────────────────────────────────────────────────────────
    conn = connect_db()
    log.info("数据库连接成功")

    # 游标：若 para_map.jsonl 已存在且有记录，从上次结束处续扫
    existing: dict[str, dict] = {}
    if map_path.exists():
        with map_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    existing[rec["html"]] = rec
        log.info("已有 para_map.jsonl，已扫描 %d 条，将跳过", len(existing))

    # 确定游标起点：已扫描的最大 para_end + 1
    if existing:
        last_para = max(v["para_end"] for v in existing.values())
        cursor = last_para + 1
        log.info("续扫游标：paragraph = %d", cursor)
    else:
        with conn.cursor() as cur:
            cur.execute("SELECT MIN(paragraph) FROM pali_texts WHERE book = %s", (book_id,))
            cursor = cur.fetchone()[0]
        log.info("游标初始化：paragraph = %d", cursor)

    # ── 主循环 ────────────────────────────────────────────────────────────────
    # 以追加模式写入，支持断点续扫
    with map_path.open("a", encoding="utf-8") as out_f:
        for html_path in html_files:
            if html_path.name in existing:
                log.info("跳过已扫描: %s", html_path.name)
                continue

            chapter_label = f"{args.corpus}/{html_path.name}"
            log.info("══════ 扫描 %s ══════", chapter_label)

            html      = html_path.read_text(encoding="utf-8")
            pali_text = extract_pali(html)

            if not pali_text.strip():
                log.warning("%s: #east 为空，跳过", chapter_label)
                continue

            try:
                matched, cursor = match_pali_texts(
                    conn, book_id, cursor, pali_text, chapter_label
                )
            except RuntimeError as e:
                log.error("致命错误: %s", e)
                conn.close()
                sys.exit(1)

            record = {
                "html":       html_path.name,
                "para_start": matched[0],
                "para_end":   matched[-1],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            log.info("  已写入: %s → 段落 %d..%d", html_path.name, matched[0], matched[-1])

    conn.close()
    log.info("扫描完成 → %s", map_path)


if __name__ == "__main__":
    main()
