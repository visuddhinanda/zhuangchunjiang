# pali-align

将汉巴对照网站的中文译文与 WikiPali `pali_sentences` 表逐句对齐，输出 JSONL 供后续导入使用。

## 项目结构

```
pali-align/
├── src/
│   ├── 01_download.py   # 从网站下载 HTML 页面
│   └── 02_align.py      # 对齐并输出 JSONL
├── html/
│   └── milinda/              # 下载的 HTML 文件（由 01_download.py 生成）
│       ├── 001.html
│       ├── 002.html
│       ├── meta.toml         # 章节元数据（自动生成）
│       └── para_map.jsonl    # HTML↔段落对照表（由 03_scan_paragraphs.py 生成）
├── jsonl/
│   └── milinda/
│       ├── 001.jsonl         # 对齐结果（由 02_align.py 生成）
│       └── 002.jsonl
├── .env                 # 数据库配置（从 .env.example 复制）
├── .env.example
├── requirements.txt
└── README.md
```

## 安装

```bash
pip install -r requirements.txt
```

复制并填写数据库配置：

```bash
cp .env.example .env
```

## 使用步骤

```
01_download.py          下载 HTML 页面
03_scan_paragraphs.py   扫描巴利文，生成 HTML↔段落对照表
02_align.py             LLM 对齐，输出 JSONL
```

### 第一步：下载 HTML

```bash
python src/01_download.py --corpus milinda --start 1 --end 36
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--corpus` | 语料库名称，决定保存目录和 URL 模板 | 必填 |
| `--start` | 起始章节编号 | `1` |
| `--end` | 结束章节编号 | 必填 |
| `--delay` | 每次请求间隔秒数，避免过快访问 | `1.0` |

下载完成后，`html/milinda/` 目录结构如下：

```
html/milinda/
├── 001.html
├── 002.html
├── ...
├── 036.html
└── meta.toml
```

`meta.toml` 内容示例：

```toml
corpus = "milinda"
book_id = 152
chapter_start = 1
chapter_end = 36
para_start = 0    # 对齐后由 02_align.py 回填
para_end = 0
```

### 第二步：扫描巴利文段落

```bash
python src/03_scan_paragraphs.py --corpus milinda

# 只扫描部分章节
python src/03_scan_paragraphs.py --corpus milinda --start 1 --end 5
```

扫描结果保存至 `html/milinda/para_map.jsonl`，每行格式：

```json
{"html": "001.html", "paragraphs": [1, 2, 3, ..., 17]}
```

支持断点续扫：若 `para_map.jsonl` 已存在，自动跳过已扫描的文件，从上次结束处继续。

### 第三步：LLM 对齐

```bash
python src/02_align.py --corpus milinda

# 只处理部分章节
python src/02_align.py --corpus milinda --start 1 --end 3
```

对齐完成后，`jsonl/milinda/` 下每个 HTML 对应一个同名 JSONL 文件：

```
jsonl/milinda/001.jsonl
jsonl/milinda/002.jsonl
...
```

每行格式：

```json
{
    "sentence_id": 12345,
    "book": 152,
    "paragraph": 24,
    "word_begin": 2,
    "word_end": 41,
    "pali": "4. Tesu sāmaṇero jambudīpe ...",
    "chinese": "在兩位中，沙彌成為閻浮提..."
}
```

同时 `meta.toml` 的 `para_start` / `para_end` 会被回填为实际对齐的段落范围。

## 对齐算法说明

### 步骤一：提取页面文本

用 Python 内置 `html.parser` 按 id 定位 `#center`（汉译）和 `#east`（巴利文）区块，深度计数处理嵌套 div，`<br>` 转换为换行。

### 步骤二：开头验证

取网站巴利文与数据库当前游标行，去标点空白后取较短者长度做宽松比对，确保两边文本来源一致后再开始逼近。

### 步骤三：单步逼近匹配 pali_texts

整页巴利文作为一个单元（不按段落编号切分），逐条加载 `pali_texts` 累计文本长度，直到与网站长度最接近为止。游标跨章节连续递增，不重置。

### 步骤四：查询 pali_sentences

根据逼近结果的段落范围，查询对应的 `pali_sentences`，按 `paragraph, word_begin` 排序。

### 步骤五：按比例切分中文

以各 `pali_sentence` 字符长度占比估算目标中文长度，按标点（`。！？；，、`）切分中文为候选片段后贪心合并，使每组长度最接近目标。分组数不一致时降级：首组取全文，其余为 `null`。

## 新增语料库

在 `01_download.py` 的 `CORPUS_CONFIG` 中添加配置即可：

```python
CORPUS_CONFIG = {
    "milinda": {
        "url_template": "https://agama.buddhason.org/Mi/Mi{n}.htm",
        "book_id": 152,
    },
    "your_corpus": {
        "url_template": "https://example.com/...",
        "book_id": 999,
    },
}
```

## 注意事项

- `chinese` 字段为 `null` 的条目表示该句子的中文切分失败（贪心分组数不一致），需人工核对
- 开头验证失败时程序会**终止退出**，需检查游标是否与网站章节顺序一致
- 建议先用较小范围（`--start 1 --end 3`）测试后再跑全本
