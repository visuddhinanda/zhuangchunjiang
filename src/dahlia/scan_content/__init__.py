import json
import sys
from pathlib import Path
from collections import Counter

# ─── 配置 ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent.parent  # 按实际项目结构调整

# ─── 入口 ────────────────────────────────────────────────────────────────────

def launch(corpus: str) -> None:
    jsonl_dir = ROOT / "jsonl" / corpus

    if not jsonl_dir.exists():
        print(f"找不到目录：{jsonl_dir}", file=sys.stderr)
        sys.exit(1)

    jsonl_files = sorted(jsonl_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"jsonl/{corpus}/ 下没有 .jsonl 文件", file=sys.stderr)
        sys.exit(1)

    counter: Counter[str] = Counter()
    total_lines = 0
    error_lines = 0

    for jsonl_path in jsonl_files:
        for lineno, raw in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"  [警告] {jsonl_path.name}:{lineno} JSON 解析失败：{e}", file=sys.stderr)
                error_lines += 1
                continue

            content = (obj.get("content") or "").rstrip()
            total_lines += 1
            if content:
                counter[content[-1]] += 1
            else:
                counter["(空)"] += 1

    # ── 输出 ──────────────────────────────────────────────────────────────────
    print(f"\ncorpus : {corpus}")
    print(f"文件数 : {len(jsonl_files)}")
    print(f"总行数 : {total_lines}（解析错误 {error_lines} 行）")
    print(f"\n末尾字符分布（共 {len(counter)} 种）：\n")
    print(f"  {'字符':^8}  {'次数':>8}  {'占比':>7}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*7}")
    for char, count in counter.most_common():
        pct = count / total_lines * 100
        label = repr(char) if char == "(空)" else char
        print(f"  {label:^8}  {count:>8,}  {pct:>6.1f}%")
    print()
