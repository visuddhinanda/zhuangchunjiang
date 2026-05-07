#!/usr/bin/env python3
"""
01_download.py
==============
从 agama.buddhason.org 下载指定经典的汉巴对照 HTML 页面，
保存至 html/{corpus}/ 目录，并生成 meta.toml 供后续对齐使用。

用法：
    python src/01_download.py --corpus milinda --start 1 --end 36
    python src/01_download.py --corpus note --start 1 --end 10 --digits 2
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import requests
import toml

# ── 日志配置 ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://agama.buddhason.org/",
    "Accept": "text/html,application/xhtml+xml",
}

# 已知经典的 URL 模板与 WikiPali book_id
# n_digits: 章节编号的位数，0 表示不补零（原样使用），>0 表示补零到指定位数
CORPUS_CONFIG = {
    "milinda": {
        "url_template": "https://agama.buddhason.org/Mi/Mi{n}.htm",
        "book_id": 152,
        "n_digits": 0,  # 不补零，原本就是 1, 2, 3...
    },
    "dn": {
        "url_template": "https://agama.buddhason.org/DN/DN{n}.htm",
        "book_id": 93,
        "n_digits": 2,
    },
    "mn": {
        "url_template": "https://agama.buddhason.org/MN/MN{n}.htm",
        "book_id": 164,
        "n_digits": 3,
    },
    "sn": {
        "url_template": "https://agama.buddhason.org/SN/SN{n}.htm",
        "book_id": 167,
        "n_digits": 4,
    },
    "an": {
        "url_template": "https://agama.buddhason.org/AN/AN{n}.htm",
        "book_id": 167,
        "n_digits": 4,
    },
    "note": {
        "url_template": "https://agama.buddhason.org/note/note{n}.htm",
        "book_id": 93,
        "n_digits": 0,  
    },
}

# 根目录：脚本所在的 src/ 上一层
ROOT = Path(__file__).resolve().parent.parent

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="下载汉巴对照 HTML 页面")
    p.add_argument("--corpus", required=True, help="语料库名称，如 milinda")
    p.add_argument("--start", type=int, default=1, help="起始章节编号")
    p.add_argument("--end", type=int, required=True, help="结束章节编号")
    p.add_argument("--delay", type=float, default=1.0, help="每次请求间隔秒数（默认 1.0）")
    p.add_argument("--digits", type=int, default=None, 
                   help="章节编号位数，0=不补零，>0=补零到指定位数，不指定则使用配置文件默认值")
    return p.parse_args()


def format_chapter_num(n: int, digits: int) -> str:
    """
    根据位数格式化章节编号
    
    Args:
        n: 章节编号
        digits: 位数，0 表示不补零，返回原数字字符串
    
    Returns:
        格式化后的字符串
    """
    if digits <= 0:
        return str(n)
    return str(n).zfill(digits)


def build_url(template: str, n: int, digits: int) -> str:
    """
    构建 URL，支持两种替换模式：
    - {n}: 使用格式化后的编号
    - {n_raw}: 使用原始编号（不补零）
    
    例如：template = "https://example.com/page{n}.htm" 配合 digits=3
          n=5 -> "https://example.com/page005.htm"
    """
    formatted_n = format_chapter_num(n, digits)
    # 为了兼容，同时也提供原始编号的替换（如果有需要）
    raw_n = str(n)
    return template.format(n=formatted_n, n_raw=raw_n)


def get_filename(n: int, digits: int) -> str:
    """生成 HTML 文件名，如 001.html, 002.html..."""
    formatted_n = format_chapter_num(n, digits)
    return f"{formatted_n}.html"


def fetch_page(url: str, delay: float) -> str:
    """抓取单页 HTML，失败时抛出异常。"""
    log.info("GET %s", url)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    # 确保编码正确：网站声明 charset=utf-8
    resp.encoding = "utf-8"
    time.sleep(delay)
    return resp.text


def save_html(out_dir: Path, filename: str, html: str) -> None:
    """将 HTML 写入文件，使用 UTF-8 编码。"""
    out_path = out_dir / filename
    out_path.write_text(html, encoding="utf-8")
    log.info("  已保存 → %s", out_path)


def write_meta(out_dir: Path, corpus: str, book_id: int, start: int, end: int, 
               n_digits: int) -> None:
    """生成 meta.toml，供 02_align.py 读取。"""
    meta = {
        "corpus": corpus,
        "book_id": book_id,
        "chapter_start": start,
        "chapter_end": end,
        "n_digits": n_digits,  # 保存位数信息供后续处理
        # para_start / para_end 由 02_align.py 在对齐后回填
        # 这里先写占位值 0
        "para_start": 0,
        "para_end": 0,
    }
    meta_path = out_dir / "meta.toml"
    with meta_path.open("w", encoding="utf-8") as f:
        toml.dump(meta, f)
    log.info("meta.toml 已写入 → %s", meta_path)


def main() -> None:
    args = parse_args()

    if args.corpus not in CORPUS_CONFIG:
        log.error("未知 corpus: %s，支持: %s", args.corpus, list(CORPUS_CONFIG))
        sys.exit(1)

    cfg = CORPUS_CONFIG[args.corpus]
    
    # 确定位数：优先使用命令行参数，否则使用配置文件
    n_digits = args.digits if args.digits is not None else cfg.get("n_digits", 0)
    
    log.info("语料: %s, 编号位数: %s %s", 
             args.corpus, 
             n_digits, 
             "(不补零)" if n_digits <= 0 else f"(补零到 {n_digits} 位)")

    out_dir = ROOT / "html" / args.corpus
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []

    for n in range(args.start, args.end + 1):
        url = build_url(cfg["url_template"], n, n_digits)
        filename = get_filename(n, n_digits)

        try:
            html = fetch_page(url, args.delay)
            save_html(out_dir, filename, html)
        except requests.HTTPError as e:
            log.error("HTTP 错误 n=%d (url=%s): %s", n, url, e)
            failed.append(n)
        except Exception as e:
            log.error("下载失败 n=%d (url=%s): %s", n, url, e)
            failed.append(n)

    # 写入 meta.toml（即使有失败也写，方便重试）
    write_meta(out_dir, args.corpus, cfg["book_id"], args.start, args.end, n_digits)

    if failed:
        log.warning("以下章节下载失败，请手动重试: %s", failed)
        sys.exit(1)
    else:
        log.info("全部 %d 章节下载完毕。", args.end - args.start + 1)


if __name__ == "__main__":
    main()