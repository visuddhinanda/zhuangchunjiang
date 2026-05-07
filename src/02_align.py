#!/usr/bin/env python3
"""
02_align.py
===========
从 html/{corpus}/ 读取已下载的 HTML 文件，与 WikiPali pali_sentences 表对齐，
结果写入 output/{corpus}/aligned.jsonl，每行一条 pali_sentence 记录。

用法：
    python src/02_align.py --corpus milinda

每行 JSONL 格式：
    {
        "sentence_id": <pali_sentences.id>,
        "book": 152,
        "paragraph": 24,
        "word_begin": 2,
        "word_end": 41,
        "pali": "4. Tesu sāmaṇero ...",
        "chinese": "在兩位中，沙彌成為..."
    }
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from html.parser import HTMLParser

import psycopg2
import psycopg2.extras
import toml
from dotenv import load_dotenv

from llm_align import llm_split

# ── 日志配置 ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 路径 ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent


# ═════════════════════════════════════════════════════════════════════════════
# HTML 解析
# ═════════════════════════════════════════════════════════════════════════════

class DivExtractor(HTMLParser):
    """
    提取指定 id 的 div 内部文本，用深度计数处理嵌套。
    <br> 转换为换行，<a> 等内联标签透明处理（保留文本）。
    """

    def __init__(self, target_id: str):
        super().__init__()
        self.target_id = target_id
        self.depth = 0          # 目标 div 内的嵌套深度
        self.in_target = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        if self.in_target:
            self.depth += 1
            if tag == "br":
                self.parts.append("\n")
        else:
            attr_dict = dict(attrs)
            if attr_dict.get("id") == self.target_id:
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
    """从 HTML 字符串中提取指定 div 的纯文本（保留换行）。"""
    parser = DivExtractor(div_id)
    parser.feed(html)
    return parser.text


# ═════════════════════════════════════════════════════════════════════════════
# 文本工具
# ═════════════════════════════════════════════════════════════════════════════

def strip_whitespace(text: str) -> str:
    """去除所有空白（含全角空格 U+3000），用于长度比对。"""
    return re.sub(r"[\s\u3000]+", "", text)


def normalize_for_head(text: str) -> str:
    """
    用于开头验证：去标点、去空白、转小写。
    保留 Unicode 字母和数字（覆盖巴利变音字母）。
    """
    text = strip_whitespace(text)
    text = re.sub(r"[^\w]", "", text, flags=re.UNICODE)
    return text.lower()


def split_by_chinese_punctuation(text: str) -> list[str]:
    """
    按中文标点 + 换行切分文本，标点保留在片段末尾。
    修复：避免整段中文被当成一个块（导致全部落入第一句）
    """
    # 先标准化换行
    text = re.sub(r"\r\n?", "\n", text.strip())

    # 按中文标点 + 换行切分
    parts = re.split(r"(?<=[。！？；])|\n+", text)

    # 去空 & 清理多余空白
    return [p.strip() for p in parts if p.strip()]


def greedy_split(chinese: str, pali_lengths: list[int]) -> list[str | None]:
    """
    根据各 pali_sentence 的长度占比，将整页中文贪心切分为等量分组。
    
    算法：
    - 计算每条 pali_sentence 的累积长度占比（位置）
    - 按中文标点切出候选片段
    - 计算每个候选片段的累积长度占比（每个片段结束时的位置）
    - 遍历巴利文位置，独立查找最接近的中文片段
    - 如果多个巴利文对应同一个中文片段，则只给第一个，后面的为 None
    """
    
    n = len(pali_lengths)
    if n == 1:
        return [chinese]
    
    # 1. 计算巴利文的累积位置（占比）
    total_pali = sum(pali_lengths)
    pali_positions = []
    cumulative = 0
    for length in pali_lengths:
        cumulative += length
        position = cumulative / total_pali
        pali_positions.append(position)
    
    log.info("巴利文累积位置 (%d个): %s", n, [round(p, 4) for p in pali_positions])
    
    # 2. 按中文标点切分候选片段
    candidates = split_by_chinese_punctuation(chinese)
    log.info("中文候选片段数: %d", len(candidates))
    
    if not candidates:
        log.warning("中文无法按标点切分，降级")
        return [chinese] + [None] * (n - 1)
    
    for i, frag in enumerate(candidates[:10]):  # 只打印前10个
        preview = frag[:50].replace('\n', '\\n')
        log.info("  片段%d: '%s...' (长度%d)", i, preview, len(frag))
    
    # 3. 计算中文候选片段的累积位置（每个片段结束时的位置）
    total_chinese = len(chinese)
    chinese_positions = []
    cumulative = 0
    for frag in candidates:
        cumulative += len(frag)
        position = cumulative / total_chinese
        chinese_positions.append(position)
    
    # 修正最后一个位置为1.0
    if chinese_positions and abs(chinese_positions[-1] - 1.0) > 0.001:
        log.warning("最后一个中文位置=%.4f，修正为1.0", chinese_positions[-1])
        chinese_positions[-1] = 1.0
    
    log.info("中文累积位置 (%d个): %s", len(chinese_positions), [round(p, 4) for p in chinese_positions])
    
    # 4. 为每个巴利文位置查找最接近的中文片段（独立查找，允许重复）
    pali_to_chinese_idx = []  # 每个巴利文对应的中文片段索引，None表示无对应
    
    for i, pali_pos in enumerate(pali_positions):
        # 找到最接近的中文位置索引
        best_idx = min(range(len(chinese_positions)), 
                       key=lambda j: abs(chinese_positions[j] - pali_pos))
        best_diff = abs(chinese_positions[best_idx] - pali_pos)
        
        pali_to_chinese_idx.append(best_idx)
        log.debug("巴利文[%d] 位置 %.4f -> 最接近的中文片段索引: %d (位置 %.4f, 差=%.4f)", 
                 i, pali_pos, best_idx, chinese_positions[best_idx], best_diff)
    
    # 5. 根据中文片段分配，合并连续相同索引的巴利文
    # 对于相同的中文索引，只有第一个巴利文获得中文，其余为 None
    groups: list[str | None] = []
    last_used_idx = -1
    current_group_chinese = ""
    
    for i, ch_idx in enumerate(pali_to_chinese_idx):
        if ch_idx != last_used_idx:
            # 新的中文片段，重新开始收集
            if current_group_chinese:
                groups.append(current_group_chinese)
            # 开始新分组：从ch_idx对应的候选片段开始收集
            # 需要找到这个索引对应的完整中文内容（可能跨越多个候选片段？）
            # 简单方案：每个巴利文最多对应一个候选片段
            current_group_chinese = candidates[ch_idx]
            last_used_idx = ch_idx
        else:
            # 同一个中文片段，当前巴利文没有对应的中文
            groups.append(None)
            # 注意：不改变 current_group_chinese 和 last_used_idx
    
    # 添加最后一个分组
    if current_group_chinese:
        groups.append(current_group_chinese)
    
    # 调整：上面的逻辑有问题，因为groups的顺序应该和巴利文一一对应
    # 重新实现：直接构建与巴利文数量相同的列表
    groups = []
    used_chinese_indices = set()
    
    for i, ch_idx in enumerate(pali_to_chinese_idx):
        if ch_idx not in used_chinese_indices:
            # 这个中文片段还未被使用
            groups.append(candidates[ch_idx])
            used_chinese_indices.add(ch_idx)
            log.debug("巴利文[%d] -> 中文片段%d (新)", i, ch_idx)
        else:
            # 这个中文片段已经被之前的巴利文用了
            groups.append(None)
            log.debug("巴利文[%d] -> None (中文片段%d已被使用)", i, ch_idx)
    
    # 6. 验证分组数
    log.debug("原始分组数: %d (目标: %d)", len(groups), n)
    
    if len(groups) != n:
        log.error("分组数不匹配: groups=%d, n=%d", len(groups), n)
        # 补全或截断
        while len(groups) < n:
            groups.append(None)
        groups = groups[:n]
    
    # 统计
    non_null_count = sum(1 for g in groups if g is not None)
    log.info("最终分组: 非空=%d, 各group长度=%s", non_null_count, 
             [len(g) if g else 0 for g in groups])
    
    return groups

# ═════════════════════════════════════════════════════════════════════════════
# 数据库
# ═════════════════════════════════════════════════════════════════════════════

def connect_db() -> psycopg2.extensions.connection:
    """从 .env 读取数据库配置，返回 psycopg2 连接。"""
    load_dotenv(ROOT / ".env")
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_DATABASE", "wikipali"),
        user=os.getenv("DB_USERNAME", "www"),
        password=os.getenv("DB_PASSWORD", ""),
    )


def load_pali_texts_batch(conn, book_id: int, from_para: int, batch: int = 5) -> list[dict]:
    """
    从 pali_texts 表顺序加载一批段落（从 from_para 开始，取 batch 行）。
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT paragraph, text
            FROM pali_texts
            WHERE book = %s AND paragraph >= %s
            ORDER BY paragraph
            LIMIT %s
            """,
            (book_id, from_para, batch),
        )
        return [dict(r) for r in cur.fetchall()]


def load_pali_sentences(conn, book_id: int, paragraphs: list[int]) -> list[dict]:
    """查询指定段落的所有 pali_sentences，按 paragraph, word_begin 排序。"""
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
# 核心对齐逻辑
# ═════════════════════════════════════════════════════════════════════════════

def match_pali_texts(
    conn,
    book_id: int,
    cursor: int,
    pali_clean: str,
    chapter_label: str,
) -> tuple[list[int], int]:
    """
    单步逼近：逐条加载 pali_texts，累计文本长度直到与网站巴利文接近。

    开头验证：去标点空白后取前 30 字符做宽松比对。
    逼近条件：累计长度 < 网站长度 * 0.95 时继续加；
              越过临界时比较加入前后差异决定是否接受。

    返回：(匹配的 paragraph 列表, 下一个 cursor 值)
    """
    # ── 开头验证 ──────────────────────────────────────────────────────────────
    rows = load_pali_texts_batch(conn, book_id, cursor, batch=1)
    if not rows:
        raise RuntimeError(f"pali_texts 已耗尽，cursor={cursor}")
 
    agama_norm = normalize_for_head(pali_clean)
    db_norm    = normalize_for_head(rows[0]["text"])
    # 取两者中较短的长度做比对，避免 db 行过短导致假失败
    head_len   = min(30, len(agama_norm), len(db_norm))
    agama_head = agama_norm[:head_len]
    db_head    = db_norm[:head_len]

    if agama_head != db_head:
        raise RuntimeError(
            f"{chapter_label} 开头验证失败！\n"
            f"  网站: [{agama_head}]\n"
            f"  数据库: [{db_head}]"
        )
    log.info("  开头验证通过: %s", agama_head)

    # ── 单步逼近（改进版：进入接近区间后继续尝试一段）────────────────────────────

    agama_len = len(strip_whitespace(pali_clean))
    accumulated = ""
    matched: list[int] = []

    prev_accumulated = ""
    prev_matched: list[int] = []

    while True:
        rows = load_pali_texts_batch(conn, book_id, cursor, batch=1)
        if not rows:
            log.warning("  pali_texts 已耗尽，cursor=%d", cursor)
            break

        row = rows[0]
        clean_row = strip_whitespace(row["text"])

        candidate = accumulated + clean_row
        candidate_len = len(candidate)
        log.debug("指针位置%d 当前缓冲区%d 目标长度%d",cursor,candidate_len,agama_len)
        # ── 未进入接近区间：继续累加 ──
        if candidate_len < agama_len:
            accumulated = candidate
            matched.append(row["paragraph"])
            cursor += 1
            continue

        # ── 进入接近区间：记录“加入前”的状态 ──
        prev_accumulated = accumulated
        prev_matched = matched.copy()

        # 先尝试加入当前段
        accumulated = candidate
        matched.append(row["paragraph"])
        cursor += 1

        # 再尝试“多看一段”
        rows_next = load_pali_texts_batch(conn, book_id, cursor, batch=1)
        if not rows_next:
            break

        next_row = rows_next[0]
        next_clean = strip_whitespace(next_row["text"])

        candidate_next = accumulated + next_clean

        # 三种状态对比：
        # 1️⃣ prev_accumulated（未加入当前段）
        # 2️⃣ accumulated（加入当前段）
        # 3️⃣ candidate_next（再多加一段）

        diff_prev = abs(len(prev_accumulated) - agama_len)
        diff_curr = abs(len(accumulated) - agama_len)
        diff_next = abs(len(candidate_next) - agama_len)

        # ── 选择最接近的状态 ──
        if diff_prev <= diff_curr and diff_prev <= diff_next:
            # 回退到“未加入当前段”
            accumulated = prev_accumulated
            matched = prev_matched
            cursor -= 1  # 回退 cursor
        elif diff_next < diff_curr:
            # 接受“多加一段”
            accumulated = candidate_next
            matched.append(next_row["paragraph"])
            cursor += 1

        # 无论如何，结束逼近
        break

    log.info(
        "  逼近完成 | 网站=%d 累计=%d 差异=%d | p%d..p%d",
        agama_len,
        len(accumulated),
        abs(len(accumulated) - agama_len),
        matched[0] if matched else 0,
        matched[-1] if matched else 0,
    )
    return matched, cursor


def process_chapter(
    conn,
    book_id: int,
    cursor: int,
    html_path: Path,
    chapter_label: str,
    use_llm: bool = False,
) -> tuple[list[dict], int]:
    """
    处理单个章节 HTML 文件，返回 (对齐结果列表, 新 cursor)。

    对齐结果列表中每个元素对应一条 pali_sentence：
        {sentence_id, book, paragraph, word_begin, word_end, pali, chinese}
    """
    html = html_path.read_text(encoding="utf-8")

    chinese_clean = extract_text(html, "center")
    pali_clean = extract_text(html, "east")

    if not chinese_clean.strip() or not pali_clean.strip():
        raise ValueError(f"{chapter_label}: #center 或 #east 为空")

    # 步骤2：开头验证 + 单步逼近
    matched_paragraphs, cursor = match_pali_texts(
        conn, book_id, cursor, pali_clean, chapter_label
    )

    # 步骤3：查询 pali_sentences
    sentences = load_pali_sentences(conn, book_id, matched_paragraphs)
    if not sentences:
        raise ValueError(f"{chapter_label}: pali_sentences 为空，段落={matched_paragraphs}")

    log.info("  pali_sentences 共 %d 条", len(sentences))

    # 步骤4：切分中文（比例算法 或 LLM）
    if use_llm:
        log.info("  使用 LLM 切分中文")
        chinese_groups = llm_split(chinese_clean, sentences)
    else:
        pali_lengths = [len(s["text"]) for s in sentences]
        chinese_groups = greedy_split(chinese_clean, pali_lengths)

    # 步骤5：组装结果
    results = []
    for i, sent in enumerate(sentences):
        results.append({
            "book": book_id,
            "paragraph": sent["paragraph"],
            "word_begin": sent["word_begin"],
            "word_end": sent["word_end"],
            "pali": sent["text"],
            "chinese": chinese_groups[i] if i < len(chinese_groups) else None,
        })

    return results, cursor


# ═════════════════════════════════════════════════════════════════════════════
# 入口
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="对齐汉巴文本，输出 JSONL")
    p.add_argument("--corpus", required=True, help="语料库名称，如 milinda")
    p.add_argument("--start", type=int, default=None, help="起始文件编号（如 1 对应 001.html），不指定则处理全部")
    p.add_argument("--end",   type=int, default=None, help="结束文件编号（如 36 对应 036.html），不指定则处理全部")
    p.add_argument("--llm", action="store_true", help="使用 LLM 进行中文切分")
    return p.parse_args()


def main() -> None:
    args = parse_args()
 
    html_dir = ROOT / "html" / args.corpus
    meta_path = html_dir / "meta.toml"
    out_dir = ROOT / "jsonl" / args.corpus
    out_dir.mkdir(parents=True, exist_ok=True)
 
    # ── 读取 meta.toml ─────────────────────────────────────────────────────────
    if not meta_path.exists():
        log.error("找不到 %s，请先运行 01_download.py", meta_path)
        sys.exit(1)
 
    meta = toml.load(meta_path)
    book_id: int = meta["book_id"]
    chapter_start: int = meta["chapter_start"]
    chapter_end: int = meta["chapter_end"]
 
    log.info(
        "corpus=%s  book_id=%d  chapters=%d..%d",
        args.corpus, book_id, chapter_start, chapter_end,
    )
 
    # ── 数据库连接 ─────────────────────────────────────────────────────────────
    conn = connect_db()
    log.info("数据库连接成功")
 
    # 游标初始化：取 book 的最小 paragraph
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MIN(paragraph) FROM pali_texts WHERE book = %s", (book_id,)
        )
        cursor: int = cur.fetchone()[0]
    log.info("游标初始化：paragraph = %d", cursor)
 
    # ── 主循环：扫描目录文件列表，有多少处理多少 ────────────────────────────────
    html_files = sorted(html_dir.glob("*.html"))
    if not html_files:
        log.error("html/%s/ 目录下没有找到任何 .html 文件", args.corpus)
        sys.exit(1)
 
    # 按 --start / --end 过滤（文件名去掉扩展名后转为整数，如 001 → 1）
    if args.start is not None or args.end is not None:
        def _file_no(p: Path) -> int:
            try:
                return int(p.stem)
            except ValueError:
                return -1
        lo = args.start if args.start is not None else 1
        hi = args.end   if args.end   is not None else 999999
        html_files = [p for p in html_files if lo <= _file_no(p) <= hi]
 
    log.info("找到 %d 个 HTML 文件（start=%s end=%s）", len(html_files), args.start, args.end)
 
    total_written = 0
    first_para = None
    last_para = None
 
    for html_path in html_files:
        chapter_label = f"{args.corpus}/{html_path.name}"
 
        if not html_path.is_file():
            continue
 
        log.info("══════ 处理 %s ══════", chapter_label)
 
        try:
            results, cursor = process_chapter(
                conn, book_id, cursor, html_path, chapter_label,
                use_llm=args.llm,
            )
        except RuntimeError as e:
            # 开头验证失败：致命错误，终止
            log.error("致命错误: %s", e)
            conn.close()
            sys.exit(1)
        except Exception as e:
            log.warning("%s 处理失败，跳过: %s", chapter_label, e)
            continue
 
        # 每个 HTML 对应一个同名 jsonl 文件
        out_path = out_dir / html_path.with_suffix(".jsonl").name
        with out_path.open("w", encoding="utf-8") as out_f:
            for record in results:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if first_para is None:
                    first_para = record["paragraph"]
                last_para = record["paragraph"]
 
        total_written += len(results)
        log.info("  写入 %d 条 → %s，累计 %d 条", len(results), out_path.name, total_written)
 
    # ── 回填 meta.toml 的段落范围 ─────────────────────────────────────────────
    if first_para is not None:
        meta["para_start"] = first_para
        meta["para_end"] = last_para
        with meta_path.open("w", encoding="utf-8") as f:
            toml.dump(meta, f)
        log.info("meta.toml 已更新：para_start=%d  para_end=%d", first_para, last_para)
 
    conn.close()
    log.info("完成，共写入 %d 条 → %s", total_written, out_path)

if __name__ == "__main__":
    main()