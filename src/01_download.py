#!/usr/bin/env python3
"""
01_download.py
==============
从 agama.buddhason.org 下载指定经典的汉巴对照 HTML 页面，
保存至 html/{corpus}/ 目录，并生成 meta.toml 供后续对齐使用。

用法：
    python src/01_download.py --corpus milinda --start 1 --end 36
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
CORPUS_CONFIG = {
    "milinda": {
        "url_template": "https://agama.buddhason.org/Mi/Mi{n}.htm",
        "book_id": 152,
    },
    "dn1": {
        "url_template": "https://agama.buddhason.org/DN/DN{n}.htm",
        "book_id": 93,
    },
     "note": {
        "url_template": "https://agama.buddhason.org/note/note{n}.htm",
        "book_id": 93,
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
    return p.parse_args()


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


def write_meta(out_dir: Path, corpus: str, book_id: int, start: int, end: int) -> None:
    """生成 meta.toml，供 02_align.py 读取。"""
    meta = {
        "corpus": corpus,
        "book_id": book_id,
        "chapter_start": start,
        "chapter_end": end,
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
    out_dir = ROOT / "html" / args.corpus
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []

    for n in range(args.start, args.end + 1):
        url = cfg["url_template"].format(n=n)
        filename = f"{n:03d}.html"  # 001.html, 002.html, ...

        try:
            html = fetch_page(url, args.delay)
            save_html(out_dir, filename, html)
        except requests.HTTPError as e:
            log.error("HTTP 错误 n=%d: %s", n, e)
            failed.append(n)
        except Exception as e:
            log.error("下载失败 n=%d: %s", n, e)
            failed.append(n)

    # 写入 meta.toml（即使有失败也写，方便重试）
    write_meta(out_dir, args.corpus, cfg["book_id"], args.start, args.end)

    if failed:
        log.warning("以下章节下载失败，请手动重试: %s", failed)
        sys.exit(1)
    else:
        log.info("全部 %d 章节下载完毕。", args.end - args.start + 1)


if __name__ == "__main__":
    main()