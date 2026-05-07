#!/usr/bin/env python3
"""
03_llm_align.py
===============
使用兼容 OpenAI REST API 的大模型，对巴利文句子列表与中文译文进行对齐。

主要对外接口：
    llm_split(chinese, sentences) -> list[str | None]

.env 配置：
    LLM_BASE_URL=https://api.openai.com/v1
    LLM_MODEL=gpt-4o
    LLM_API_KEY=sk-...
"""

import json
import logging
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# ── 加载 .env ────────────────────────────────────────────────────────────────
load_dotenv(ROOT / ".env")

LLM_URL     = os.getenv("LLM_URL", "https://api.openai.com/v1/chat/completions")
LLM_MODEL   = os.getenv("LLM_MODEL", "gpt-4o")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# ── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
你是一位巴利文与汉文文本对齐专家。
我会给你一段汉文译文，以及对应的巴利文句子列表（每条有序号）。
请将汉文译文切分并分配给各条巴利文句子。

输出规则：
1. 每行输出一条 JSON，格式为 {"seq": <序号>, "chinese": "<对应汉文>"}
2. 序号必须连续覆盖所有巴利文句子，不得遗漏
3. 如果某条巴利文句子无法找到对应汉文，输出 {"seq": <序号>, "chinese": null}
4. 汉文内容不要重复出现在多个序号中
5. 不要输出任何其他文字，只输出 JSONL
"""

def _build_user_prompt(chinese: str, sentences: list[dict]) -> str:
    """
    构建发送给 LLM 的用户消息。

    sentences 中每个元素包含：paragraph, word_begin, word_end, text（巴利文）
    """
    pali_lines = "\n".join(
        f"{i + 1}. {s['text']}" for i, s in enumerate(sentences)
    )
    return (
        f"【巴利文句子列表】\n{pali_lines}\n\n"
        f"【汉文译文】\n{chinese.strip()}"
    )


# ── 流式请求 ─────────────────────────────────────────────────────────────────

def _stream_chat(user_prompt: str) -> list[dict]:
    """
    向 LLM 发送流式请求，逐行解析 JSONL 输出。

    每收到完整的一行立即解析并记录日志，方便实时监控。
    返回已解析的 dict 列表：[{"seq": 1, "chinese": "..."}, ...]
    """
    url     = LLM_URL
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":    LLM_MODEL,
        "stream":   True,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    }

    log.info("  [LLM] 发送请求 model=%s url=%s", LLM_MODEL, LLM_URL)

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as resp:
        resp.raise_for_status()

        raw_lines: list[dict] = []
        buf = ""  # 跨 chunk 的行缓冲区

        for chunk in resp.iter_lines(decode_unicode=False):
            # iter_lines 返回 bytes，手动 UTF-8 解码避免 requests 猜错编码
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")
            # SSE 格式：每行形如 "data: {...}" 或 "data: [DONE]"
            if not chunk or not chunk.startswith("data:"):
                continue

            data_str = chunk[len("data:"):].strip()
            if data_str == "[DONE]":
                break

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            delta = data.get("choices", [{}])[0].get("delta", {})
            token = delta.get("content", "")
            if not token:
                continue

            buf += token

            # 检测是否有完整行（以换行符分隔）
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                parsed = _try_parse_jsonl_line(line)
                if parsed is not None:
                    raw_lines.append(parsed)
                    log.info(
                        "  [LLM] seq=%s chinese=%s",
                        parsed.get("seq"),
                        str(parsed.get("chinese", ""))[:40],
                    )
                else:
                    log.warning("  [LLM] 无法解析行: %s", line[:120])

        # 处理末尾没有换行的最后一行
        if buf.strip():
            parsed = _try_parse_jsonl_line(buf.strip())
            if parsed is not None:
                raw_lines.append(parsed)
                log.info(
                    "  [LLM] seq=%s chinese=%s",
                    parsed.get("seq"),
                    str(parsed.get("chinese", ""))[:40],
                )

    log.info("  [LLM] 流结束，共解析 %d 行", len(raw_lines))
    return raw_lines


def _try_parse_jsonl_line(line: str) -> dict | None:
    """
    尝试将一行字符串解析为 JSON dict。
    LLM 偶尔会在 JSON 前后加 ``` 或其他修饰，做简单清理后再解析。
    """
    # 去除 markdown 代码块标记
    line = re.sub(r"^```[a-z]*", "", line).strip("`").strip()
    try:
        obj = json.loads(line)
        if isinstance(obj, dict) and "seq" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    return None


# ── 主接口 ───────────────────────────────────────────────────────────────────

def llm_split(chinese: str, sentences: list[dict]) -> list[str | None]:
    """
    使用 LLM 将整页中文按巴利文句子列表切分。

    参数：
        chinese   - 整页汉文译文
        sentences - pali_sentences 查询结果列表，每项含 paragraph/word_begin/word_end/text

    返回：
        与 sentences 等长的列表，每项为对应的汉文字符串或 None（对齐失败）
    """
    n = len(sentences)
    if n == 0:
        return []

    user_prompt = _build_user_prompt(chinese, sentences)
    raw_lines   = _stream_chat(user_prompt)

    # 按 seq 建立映射
    seq_map: dict[int, str | None] = {}
    for item in raw_lines:
        seq = item.get("seq")
        if isinstance(seq, int):
            seq_map[seq] = item.get("chinese")  # 可能为 null → None

    # 按句子顺序组装结果（seq 从 1 开始）
    result: list[str | None] = []
    for i in range(n):
        seq = i + 1
        if seq in seq_map:
            result.append(seq_map[seq])
        else:
            log.warning("  [LLM] seq=%d 无对应输出，填充 None", seq)
            result.append(None)

    non_null = sum(1 for r in result if r is not None)
    log.info("  [LLM] 对齐完成：%d/%d 条有内容", non_null, n)
    return result
