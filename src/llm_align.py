#!/usr/bin/env python3
"""
llm_align.py
============
使用兼容 OpenAI REST API 的大模型，对巴利文句子列表与中文译文进行对齐。

主要对外接口：
    llm_split(chinese, sentences) -> (list[str | None], LlmUsage)

.env 配置：
    LLM_URL=https://api.openai.com/v1/chat/completions
    LLM_MODEL=gpt-4o
    LLM_API_KEY=sk-...
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
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

# ── 用量数据结构 ──────────────────────────────────────────────────────────────

@dataclass
class LlmUsage:
    model:             str = ""
    prompt_tokens:     int = 0
    completion_tokens: int = 0
    total_tokens:      int = 0
    extra:             dict = field(default_factory=dict)  # 厂商扩展字段


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
    pali_lines = "\n".join(
        f"{i + 1}. {s['text']}" for i, s in enumerate(sentences)
    )
    return (
        f"【巴利文句子列表】\n{pali_lines}\n\n"
        f"【汉文译文】\n{chinese.strip()}"
    )


# ── 流式请求 ─────────────────────────────────────────────────────────────────

def _stream_chat(user_prompt: str) -> tuple[list[dict], LlmUsage]:
    """
    向 LLM 发送流式请求，逐行解析 JSONL 输出。
    返回 (已解析行列表, LlmUsage)。

    usage 从流末尾的 [DONE] 前最后一个 chunk 读取（stream_options.include_usage）。
    若 API 不支持，则 tokens 均为 0。
    """
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":   LLM_MODEL,
        "stream":  True,
        # 请求在流末尾附加 usage chunk（OpenAI / 兼容厂商均支持）
        "stream_options": {"include_usage": True},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    }

    log.info("  [LLM] 发送请求 model=%s url=%s", LLM_MODEL, LLM_URL)

    usage     = LlmUsage(model=LLM_MODEL)
    raw_lines: list[dict] = []
    buf = ""

    with requests.post(LLM_URL, headers=headers, json=payload, stream=True, timeout=120) as resp:
        resp.raise_for_status()

        for chunk in resp.iter_lines(decode_unicode=False):
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")
            if not chunk or not chunk.startswith("data:"):
                continue

            data_str = chunk[len("data:"):].strip()
            if data_str == "[DONE]":
                break

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # ── 读取 usage（出现在末尾的专用 chunk 或普通 chunk 里）────────────
            if "usage" in data and data["usage"]:
                u = data["usage"]
                usage.prompt_tokens     = u.get("prompt_tokens", 0)
                usage.completion_tokens = u.get("completion_tokens", 0)
                usage.total_tokens      = u.get("total_tokens", 0)
                # 保留厂商扩展字段（如 prompt_cache_hit_tokens 等）
                usage.extra = {
                    k: v for k, v in u.items()
                    if k not in {"prompt_tokens", "completion_tokens", "total_tokens"}
                }

            # ── 读取文本 delta ─────────────────────────────────────────────────
            delta = data.get("choices", [{}])[0].get("delta", {})
            token = delta.get("content", "")
            if not token:
                continue

            buf += token

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
                    log.debug("  [LLM] 跳过非JSON行: %s", line[:80])

        # 末尾无换行的最后一行
        if buf.strip():
            parsed = _try_parse_jsonl_line(buf.strip())
            if parsed is not None:
                raw_lines.append(parsed)
                log.info(
                    "  [LLM] seq=%s chinese=%s",
                    parsed.get("seq"),
                    str(parsed.get("chinese", ""))[:40],
                )

    log.info(
        "  [LLM] 流结束，共解析 %d 行 | tokens: prompt=%d completion=%d total=%d",
        len(raw_lines),
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
    )
    return raw_lines, usage


def _try_parse_jsonl_line(line: str) -> dict | None:
    line = re.sub(r"^```[a-z]*", "", line).strip("`").strip()
    try:
        obj = json.loads(line)
        if isinstance(obj, dict) and "seq" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    return None


# ── 主接口 ───────────────────────────────────────────────────────────────────

def llm_split(
    chinese: str,
    sentences: list[dict],
) -> tuple[list[str | None], LlmUsage]:
    """
    使用 LLM 将整页中文按巴利文句子列表切分。

    返回：
        (groups, usage)
        groups  - 与 sentences 等长，每项为汉文字符串或 None
        usage   - LlmUsage，含 token 用量和模型信息
    """
    n = len(sentences)
    if n == 0:
        return [], LlmUsage(model=LLM_MODEL)

    user_prompt       = _build_user_prompt(chinese, sentences)
    raw_lines, usage  = _stream_chat(user_prompt)

    seq_map: dict[int, str | None] = {}
    for item in raw_lines:
        seq = item.get("seq")
        if isinstance(seq, int):
            seq_map[seq] = item.get("chinese")

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
    return result, usage
