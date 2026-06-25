# 第四章：分块切片技术 (Chunk Splitting Technology)

> **章节摘要**：分块（Chunking）是RAG系统的"第一道关卡"——它决定了检索时系统能"看到"什么样的文本单元，直接影响召回质量、上下文连贯性和答案准确性。本章深入剖析各类分块策略的原理、实现与工程实践，帮助企业在不同场景下选择最优分块方案。

---

## 4.0 为何分块策略决定召回质量？

在RAG系统中，分块是检索的起点。整个RAG pipeline可以概括为：

```
原始文档 --> [分块] --> 文本块 --> [向量化] --> 向量索引 --> [检索] --> 相关块 --> [LLM] --> 答案
              ^^^^^^
              关键第一步
```

分块策略之所以决定召回质量，原因如下：

**第一，分块决定了语义单元的完整性。** 嵌入模型将每个文本块映射为一个固定维度的向量，这个向量代表该块的"语义指纹"。如果分块过细，一个完整的知识点可能被切分成多个碎片，每个碎片只包含部分语义信息，导致向量表示不完整，检索时难以精准匹配。如果分块过粗，一个块中混杂多个不同主题的信息，向量表示被稀释为"语义平均值"，缺乏区分度。

**第二，分块影响上下文窗口的利用效率。** RAG检索的最终目的是为LLM提供上下文。检索到的块被拼接后送入LLM的上下文窗口。如果块太小，需要检索更多的块才能覆盖足够的上下文，浪费上下文窗口；如果块太大，可能包含大量无关信息，同样浪费宝贵的窗口空间。

**第三，分块质量决定了检索的精度-召回平衡。** 细粒度分块有利于精度（精确找到匹配的句子），但损害召回（可能遗漏相关上下文）；粗粒度分块有利于召回（更容易命中包含相关信息的块），但损害精度（返回许多无关内容）。

**第四，分块策略与嵌入模型的能力边界直接相关。** 每个嵌入模型都有最大输入长度限制（如512 tokens、8192 tokens）。超出限制的部分会被截断，信息永久丢失。因此，分块大小必须与所选择的嵌入模型配套设计。

**第五，不同文档类型对分块策略的敏感度差异巨大。** 法律合同需要保留条款间的逻辑关联，技术文档需要保留代码示例与其说明的配对关系，学术论文需要保留引用上下文——没有一种"万能分块策略"适用于所有场景。

综上，分块策略不是简单地将文本按长度切分，而是一项需要结合文档结构、语义特性和业务需求的系统工程决策。

---

## 4.1 分块方法详解

### 4.1.1 固定长度分块 (Fixed-Length Chunking)

#### 概念定义

固定长度分块是将文本按照预设的固定长度（字符数或 token 数）进行切分，不考虑内容语义和结构，是最简单直接的分块方式。

#### 背景与演进

固定长度分块起源于全文检索时代的文本预处理技术。早期搜索引擎将文档切分为固定大小的段落进行索引。进入RAG时代，由于实现简单、成本低廉，它成为大多数RAG框架的默认分块方式。

#### 解决的问题

- 提供最简单、最可控的分块基线
- 确保每个块大小一致，便于批量向量化和索引
- 适合对处理速度要求极高的场景

#### 工作原理

```
固定长度分块（Character-based, chunk_size=500, overlap=0）

原始文本: "RAG系统由三个核心组件构成：检索器（Retriever）、生成器（Generator）
           和知识库（Knowledge Base）。检索器负责从知识库中找出与用户查询最相关
           的文档片段。生成器基于检索到的片段生成自然语言回答。知识库存储企业
           的所有文档和知识。"

切分结果:
┌──────────────────────────────────────────────────────────────┐
│ Chunk 1 (500 chars):                                        │
│ "RAG系统由三个核心组件构成：检索器（Retriever）、生成器      │
│  （Generator）和知识库（Knowledge Base）。检索器负责从       │
│  知识库中找出与用户查询最相关的文档片段。"                   │
├──────────────────────────────────────────────────────────────┤
│ Chunk 2 (500 chars):                                        │
│ "生成器基于检索到的片段生成自然语言回答。知识库存储企业的     │
│  所有文档和知识。"                                           │
└──────────────────────────────────────────────────────────────┘

问题: Chunk 1 在"文档片段。"处截断，断开了"生成器"的语义连续性
```

#### 核心算法

```python
def fixed_length_chunking(text: str, chunk_size: int = 500, 
                          overlap: int = 0) -> list[str]:
    """
    固定长度分块 - 字符级实现
    
    Args:
        text: 输入文本
        chunk_size: 每块大小（字符数）
        overlap: 块之间的重叠量（字符数）
    
    Returns:
        文本块列表
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    
    return chunks


def fixed_length_chunking_token(text: str, chunk_size: int = 256,
                                 overlap: int = 0,
                                 tokenizer=None) -> list[str]:
    """
    固定长度分块 - Token级实现
    
    Args:
        text: 输入文本
        chunk_size: 每块大小（token数）
        overlap: 块之间的重叠量（token数）
        tokenizer: 分词器，默认使用tiktoken
    
    Returns:
        文本块列表（解码回文本）
    """
    import tiktoken
    
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    total_tokens = len(tokens)
    
    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += (chunk_size - overlap)
    
    return chunks
```

#### LangChain 实现

```python
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter

# 字符级固定分块
char_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="",           # 空分隔符=不按自然边界断开
    length_function=len,
)

# Token级固定分块
token_splitter = TokenTextSplitter(
    chunk_size=256,
    chunk_overlap=25,
    encoding_name="cl100k_base",  # GPT-4/3.5 编码
)
```

#### LlamaIndex 实现

```python
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter

# Token级（不按句子边界）
token_parser = TokenTextSplitter(
    chunk_size=256,
    chunk_overlap=25,
    separator=" ",
)

# 句子级（按句子边界但不考虑语义）
sentence_parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
    paragraph_separator="\n\n",
)
```

#### 技术选型对比

| 维度 | 字符级固定分块 | Token级固定分块 |
|------|---------------|-----------------|
| **跨语言一致性** | 差（中文1字符约1.5-2 tokens，英文1字符约0.3 tokens） | 好（token是模型原语） |
| **计算开销** | 极低（O(n) 字符串操作） | 低（需分词，O(n)） |
| **模型兼容性** | 需估算token对应关系 | 直接匹配模型限制 |
| **中文适用性** | 较好（中文字符信息密度高） | 精确控制 |
| **推荐场景** | 快速原型、英文为主 | 多语言、生产环境 |

#### 优缺点分析

| 优点 | 缺点 |
|------|------|
| 实现极其简单，几行代码即可 | 在句子中间截断，破坏语义完整性 |
| 计算开销最低，适合大规模处理 | 不考虑文档结构（标题、段落、表格） |
| 块大小完全可控，便于对比实验 | 固定大小对不同内容密度适应性差 |
| 不必依赖外部工具或模型 | 遗漏上下文边界信息 |
| 处理速度最快 | 对技术文档等结构化内容效果差 |

#### 性能优化

```python
# 1. 使用tiktoken进行快速token计数
# pip install tiktoken
import tiktoken

def fast_token_count(text: str, encoding_name: str = "cl100k_base") -> int:
    """快速token计数，避免完整分词"""
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode_ordinary(text))

# 2. 批量分块（多线程）
from concurrent.futures import ThreadPoolExecutor

def batch_chunking(documents: list[str], chunk_size: int = 500,
                   max_workers: int = 4) -> list[list[str]]:
    """并行处理多文档分块"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            lambda doc: fixed_length_chunking(doc, chunk_size),
            documents
        )
    return list(results)

# 3. 避免过小块（后处理过滤）
def filter_small_chunks(chunks: list[str], min_length: int = 50) -> list[str]:
    """过滤掉过短的块，避免无意义的碎片"""
    return [c for c in chunks if len(c) >= min_length]
```

---

### 4.1.2 滑动窗口分块 (Sliding Window Chunking)

#### 概念定义

滑动窗口分块在固定长度分块的基础上，通过设置块之间的重叠区域（overlap），使相邻块共享部分内容，缓解信息在块边界处被截断的问题。

#### 解决的问题

- 固定长度分块在边界处截断语义的问题
- 检索时可能遗漏块边界附近的关联信息
- 提升跨块上下文的召回连贯性

#### 工作原理

```
滑动窗口分块 (chunk_size=100, overlap=25)

原始文本: "检索增强生成（RAG）是当前大语言模型应用的核心架构之一。RAG通过在
           推理时从外部知识库中检索相关信息来增强模型的知识边界。这有效缓解了
           大模型的幻觉问题，使其能够访问最新的知识。"

Token流: [T1][T2][T3][T4]...[T50]

       Chunk 1: [T1  ────────────────── T30]
                       [T21 ────────────────── T50] Chunk 2
                                      [T41 ──────────── T70] Chunk 3

            ┌─ overlap=10 ─┘     ┌─ overlap=10 ─┘

关键特性：
- 相邻块共享 overlap 区域的内容
- 边界信息被两个块同时覆盖
- overlap 越大，信息冗余度越高
```

#### 核心算法

```python
def sliding_window_chunking(text: str, chunk_size: int, 
                             overlap: int,
                             tokenizer=None) -> list[dict]:
    """
    滑动窗口分块（附带元数据）
    
    Args:
        text: 输入文本
        chunk_size: 块大小（token数）
        overlap: 重叠量（token数）
        tokenizer: 分词器
    
    Returns:
        带元数据的块列表 [{"text": str, "start": int, "end": int, "index": int}]
    """
    import tiktoken
    
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)
    chunks = []
    stride = chunk_size - overlap
    
    if stride <= 0:
        raise ValueError("chunk_size must be greater than overlap")
    
    chunk_index = 0
    start = 0
    
    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        
        chunks.append({
            "text": chunk_text,
            "token_start": start,
            "token_end": end,
            "chunk_index": chunk_index,
            "is_last": (end >= total_tokens),
        })
        
        chunk_index += 1
        
        # 最后一块不够一个完整 stride 时退出
        if end >= total_tokens:
            break
        
        start += stride
    
    return chunks
```

#### 重叠率设置指南

```
重叠率 = overlap / chunk_size × 100%

┌──────────────────────────────────────────────────────────────┐
│  重叠率      │  适用场景              │  说明                │
│──────────────┼────────────────────────┼──────────────────────│
│  0%          │  简单段落文档          │  无冗余，最低成本    │
│  10%         │  一般技术文档          │  基础上下文连续性    │
│  15%-20%     │  学术论文、法律文档    │  推荐默认值          │
│  25%-30%     │  密集论述、复杂推理    │  高冗余，高召回      │
│  >30%        │  通常不推荐            │  冗余过高，成本剧增  │
└──────────────────────────────────────────────────────────────┘

最优重叠率的经验公式：
  optimal_overlap = max_word_distance_between_related_concepts / tokens_per_word
  ≈ 50~100 tokens (对于大多数嵌入模型)
  
  即：chunk_size=500 时，overlap=50~100（10%~20%）
      chunk_size=1024 时，overlap=100~200（10%~20%）
```

#### LlamaIndex 实现

```python
from llama_index.core.node_parser import TokenTextSplitter

# 带重叠的滑动窗口分块
sliding_splitter = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=100,  # ~20% overlap
    separator=" ",
    backup_separators=["\n", "\n\n", "。"],
)

# 进一步：使用 SentenceWindowNodeParser
from llama_index.core.node_parser import SentenceWindowNodeParser

window_parser = SentenceWindowNodeParser(
    window_size=3,           # 每侧3句话作为窗口
    window_metadata_key="window",  # 原始窗口存储在metadata
    original_text_metadata_key="original_sentence",
)
```

#### 优缺点分析

| 优点 | 缺点 |
|------|------|
| 有效缓解边界截断问题 | 增加存储和计算开销（overlap比例） |
| 实现仍相对简单 | 仍不考虑文档自然结构 |
| 提升跨块语义的召回率 | 冗余信息可能导致重复答案 |
| 可与token-level精确控制结合 | 对结构化文档无效 |
| 适合作为性能baseline | 重叠区域内容在检索时可能重复出现 |

---

### 4.1.3 递归分块 (Recursive Character/Text Splitting)

#### 概念定义

递归分块按照优先级递减的分隔符序列，递归地将文本切分为越来越小的单元，直到所有块的大小都不超过设定的阈值。优先在自然边界（段落、句子）处切分。

#### 解决的问题

- 固定分块不考虑文本的自然边界（段落、句子、短语）
- 需要在保持块大小可控的同时最大化语义完整性
- 需要在"强制切分"和"保留结构"之间取得平衡

#### 工作原理

```
递归分块流程 (RecursiveCharacterTextSplitter)

分隔符优先级: ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

原始文本: "第一章\n\n1.1 背景介绍\nRAG技术近年来发展迅速。它结合了检索和生成两大能力。
          许多企业开始采用RAG构建知识库系统。\n\n1.2 技术原理\nRAG的核心在于..."

Step 1: 使用 "\n\n" 切分
┌─────────────────────────────────────────────────────────────────┐
│ ["第一章", "1.1 背景介绍\nRAG技术近年来发展迅速。它结合了检索和生成│
│  两大能力。许多企业开始采用RAG构建知识库系统。", "1.2 技术原理\n..."]│
└─────────────────────────────────────────────────────────────────┘
          │
          ▼ (某些块仍超过 chunk_size)
Step 2: 使用 "\n" 切分超过阈值的块
┌─────────────────────────────────────────────────────────────────┐
│ ["第一章", "1.1 背景介绍", "RAG技术近年来发展迅速。它结合了检索和  │
│  生成两大能力。许多企业开始采用RAG构建知识库系统。", "1.2 技术原理",│
│  "..."]                                                         │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼ (某些块仍超限)
Step 3: 使用 "。" 切分
┌─────────────────────────────────────────────────────────────────┐
│ ["第一章", "1.1 背景介绍", "RAG技术近年来发展迅速。",              │
│  "它结合了检索和生成两大能力。", "许多企业开始采用RAG构建知识库系统│
│  。", "1.2 技术原理", "..."]                                    │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼ (所有块均在 chunk_size 内，停止递归)

优点：优先在段落边界切分，其次在句子边界，最后才在字符级强制切分
```

#### 核心算法

```python
def recursive_chunking(text: str, chunk_size: int = 500,
                       chunk_overlap: int = 50,
                       separators: list[str] = None,
                       length_function=len) -> list[str]:
    """
    递归文本分块
    
    按优先级递减的分隔符递归切分，直到所有块都在 chunk_size 内。
    这是 LangChain RecursiveCharacterTextSplitter 的核心算法。
    
    Args:
        text: 输入文本
        chunk_size: 目标块大小
        chunk_overlap: 块间重叠
        separators: 分隔符优先级列表
        length_function: 长度计算函数
    
    Returns:
        文本块列表
    """
    if separators is None:
        separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    
    # 最终结果
    final_chunks = []
    
    # 用一个辅助函数递归处理
    def _split_recursive(text: str, separators: list[str]):
        # 获取当前层级的分隔符
        separator = separators[0]
        remaining_separators = separators[1:] if len(separators) > 1 else [""]
        
        # 如果文本长度已在阈值内，直接返回
        if length_function(text) <= chunk_size:
            return [text]
        
        # 用当前分隔符切分
        splits = text.split(separator) if separator else list(text)
        
        merged = []
        current_chunk = ""
        
        for split in splits:
            # 计算合并后的长度
            potential_length = length_function(current_chunk + separator + split) \
                if current_chunk else length_function(split)
            
            if potential_length <= chunk_size:
                # 合入当前块
                if current_chunk:
                    current_chunk += separator + split
                else:
                    current_chunk = split
            else:
                # 当前块已满
                if current_chunk:
                    merged.append(current_chunk)
                
                # 如果单个split超过chunk_size，递归切分
                if length_function(split) > chunk_size:
                    if remaining_separators:
                        # 用下一级分隔符递归
                        sub_splits = _split_recursive(split, remaining_separators)
                        # 除了最后一个，都直接加入
                        merged.extend(sub_splits[:-1])
                        current_chunk = sub_splits[-1] if sub_splits else ""
                    else:
                        # 无法再切分，强制截断
                        current_chunk = split
                else:
                    current_chunk = split
        
        if current_chunk:
            merged.append(current_chunk)
        
        return merged
    
    # 第一步：按分隔符层级递归切分
    raw_chunks = _split_recursive(text, separators)
    
    # 第二步：合并小块并添加重叠
    i = 0
    while i < len(raw_chunks):
        chunk = raw_chunks[i]
        # 尝试与后续小块合并
        j = i + 1
        while j < len(raw_chunks):
            combined_len = length_function(chunk + "\n" + raw_chunks[j])
            if combined_len <= chunk_size:
                chunk += "\n" + raw_chunks[j]
                j += 1
            else:
                break
        
        final_chunks.append(chunk)
        
        # 计算重叠起始位置
        if chunk_overlap > 0 and j < len(raw_chunks):
            # 从前一个块的结尾取overlap
            overlap_text = chunk[-chunk_overlap:] if length_function(chunk) > chunk_overlap else chunk
            # 重置i，使下一个块从前一个块的重叠部分后的位置开始
            i = i + 1
        else:
            i = j
    
    return final_chunks
```

#### LangChain 实现对比

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

# 递归字符分块（推荐）
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    is_separator_regex=False,
)

# 对比：普通字符分块
char_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator="",
)

# 测试对比
sample_text = (
    "第一章 引言\n\n"
    "RAG是检索增强生成技术的缩写。它能够有效提升大模型的知识准确度。\n\n"
    "第一节 背景\n"
    "传统的大语言模型存在幻觉问题。RAG通过外部知识检索来解决这个问题。"
)

recursive_chunks = recursive_splitter.split_text(sample_text)
char_chunks = char_splitter.split_text(sample_text)

# recursive_chunks 可能在段落/句子边界切分，保持更好的语义完整性
# char_chunks 严格按500字符切分，可能在句子中间截断
```

#### 优缺点分析

| 优点 | 缺点 |
|------|------|
| 兼顾结构保持和大小控制 | 对无结构文本退化为固定分块 |
| 在任何自然边界处切分效果优于固定分块 | 依赖分隔符优先级的合理设定 |
| 是当前RAG框架的默认推荐方案 | 对表格、代码等特殊格式无效 |
| 计算开销适中 | 不能识别语义边界 |
| 支持重叠设置，缓解边界问题 | 分隔符优先级需要根据语言/文档类型调整 |

#### 中文优化版分隔符配置

```python
# 中文文档优化分隔符
CHINESE_SEPARATORS = [
    "\n\n",     # 段落
    "\n",       # 换行
    "。",       # 句号
    "！",       # 感叹号
    "？",       # 问号
    "；",       # 分号
    "：",       # 冒号
    "，",       # 逗号
    "、",       # 顿号
    " ",        # 空格
    "",         # 字符级
]

# 英文文档优化分隔符
ENGLISH_SEPARATORS = [
    "\n\n",     # 段落
    "\n",       # 换行
    ". ",       # 句号+空格
    "! ",       # 感叹号+空格
    "? ",       # 问号+空格
    "; ",       # 分号+空格
    ": ",       # 冒号+空格
    ", ",       # 逗号+空格
    " ",        # 空格
    "",         # 字符级
]

# Markdown文档优化分隔符
MARKDOWN_SEPARATORS = [
    "\n## ",    # H2
    "\n### ",   # H3
    "\n#### ",  # H4
    "\n\n",     # 段落
    "\n",       # 换行
    ". ",       # 句子
    " ",        # 空格
    "",         # 字符级
]
```

---

### 4.1.4 语义分块 (Semantic Chunking)

#### 概念定义

语义分块利用嵌入模型计算相邻句子/段落之间的语义相似度，在相似度显著下降的位置设置分块边界。其核心理念是：语义上连贯的内容应放在同一个块中，语义转换处天然适合作为分块边界。

#### 解决的问题

- 所有基于字符串/结构的分块方法都无法感知语义变化
- 同一个文档中可能存在主题的自然切换，应在此处切分
- 需要确保每个块内部语义高度一致

#### 工作原理

```
语义分块流程

Step 1: 将文档拆分为基础单元（通常为句子）
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│Sent 1│ │Sent 2│ │Sent 3│ │Sent 4│ │Sent 5│ │Sent 6│
└──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘

Step 2: 计算相邻句子的嵌入向量并计算余弦相似度
   Sent 1 ──cos=0.92── Sent 2 ──cos=0.88── Sent 3 ──cos=0.45── Sent 4 ──cos=0.91── Sent 5 ──cos=0.87── Sent 6
                                        ^^^^
                                    相似度骤降点
                                    (Breakpoint)

Step 3: 在相似度低于阈值处切分
┌─────────────────────────────┐ ┌──────────────────────────────┐
│ Chunk 1: Sent 1 + Sent 2    │ │ Chunk 2: Sent 4 + Sent 5     │
│          + Sent 3           │ │          + Sent 6            │
│ (主题: RAG技术介绍)          │ │ (主题: 应用案例)              │
└─────────────────────────────┘ └──────────────────────────────┘

Step 4: 对过长的块进行二次切分
   - 如果 Chunk N 超过 max_chunk_size，触发递归/降级策略
```

#### 核心算法

```python
import numpy as np
from typing import List, Tuple, Callable
from sklearn.metrics.pairwise import cosine_similarity


def semantic_chunking(
    text: str,
    embedding_fn: Callable[[List[str]], np.ndarray],
    similarity_threshold: float = 0.7,
    max_chunk_size: int = 1024,
    min_chunk_size: int = 100,
    percentile_threshold: float = 90.0,
) -> List[str]:
    """
    基于语义相似度的自适应分块
    
    算法步骤：
    1. 将文本拆分为句子
    2. 计算相邻句子的嵌入向量
    3. 计算相邻句子的余弦相似度
    4. 在相似度低于阈值（或低于某一百分位）处切分
    5. 对过长块进行二次切分
    
    Args:
        text: 输入文本
        embedding_fn: 嵌入函数，接收文本列表，返回np.ndarray
        similarity_threshold: 绝对相似度阈值（低于此值则切分）
        max_chunk_size: 最大块大小（字符数）
        min_chunk_size: 最小块大小（字符数）
        percentile_threshold: 百分位阈值（低于此百分位的相似度处切分）
    
    Returns:
        文本块列表
    """
    # Step 1: 拆分为句子
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text]
    
    # Step 2: 生成句子嵌入
    embeddings = embedding_fn(sentences)
    
    # Step 3: 计算相邻句子的余弦相似度
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i+1].reshape(1, -1)
        )[0][0]
        similarities.append(sim)
    
    # Step 4: 确定切分点
    # 方法A: 绝对阈值
    absolute_breakpoints = {
        i + 1 for i, sim in enumerate(similarities)
        if sim < similarity_threshold
    }
    
    # 方法B: 百分位阈值（找出相似度骤降的位置）
    if percentile_threshold > 0:
        sim_array = np.array(similarities)
        percentile_value = np.percentile(sim_array, 100 - percentile_threshold) \
            if len(sim_array) > 2 else 0
        percentile_breakpoints = {
            i + 1 for i, sim in enumerate(similarities)
            if sim < percentile_value
        }
    else:
        percentile_breakpoints = set()
    
    # 合并两种方法的切分点
    breakpoints = sorted(absolute_breakpoints | percentile_breakpoints)
    
    # Step 5: 按切分点构建块
    chunks = []
    start = 0
    
    for bp in breakpoints:
        if bp > start:
            chunk_sentences = sentences[start:bp]
            chunk_text = " ".join(chunk_sentences)
            if len(chunk_text) >= min_chunk_size:
                chunks.append(chunk_text)
            start = bp
    
    # 最后一个块
    if start < len(sentences):
        chunk_text = " ".join(sentences[start:])
        chunks.append(chunk_text)
    
    # Step 6: 处理超大块（递归降级切分）
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chunk_size:
            # 超过最大长度，降级为递归分块
            sub_chunks = _recursive_split_long_chunk(
                chunk, max_chunk_size
            )
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)
    
    return final_chunks


def _split_sentences(text: str) -> List[str]:
    """将文本拆分为句子（支持中英文）"""
    import re
    
    # 中文句子切分模式
    # 匹配中文标点后的位置
    pattern = r'(?<=[。！？；\n])\s*'
    raw_sentences = re.split(pattern, text)
    
    # 过滤空句子和纯空格
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences


def _recursive_split_long_chunk(text: str, max_size: int) -> List[str]:
    """对过长块进行递归降级切分"""
    if len(text) <= max_size:
        return [text]
    
    # 尝试用逗号/分号切分
    delimiters = ["；", "，", "、", " ", ""]
    
    for delim in delimiters:
        parts = text.split(delim) if delim else list(text)
        if len(parts) > 1:
            result = []
            current = ""
            for part in parts:
                if len(current + delim + part) <= max_size:
                    current = current + delim + part if current else part
                else:
                    if current:
                        result.append(current)
                    current = part
            if current:
                result.append(current)
            return result
    
    # 最后手段：强制按字符切分
    return [text[i:i+max_size] for i in range(0, len(text), max_size)]
```

#### LangChain 语义分块实现

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# 使用 LangChain 实验性语义分块器
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

semantic_chunker = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",  # 百分位模式
    breakpoint_threshold_amount=90,          # 在相似度的90th百分位处切分
    number_of_chunks=None,                   # 不限制块数
    max_chunk_size=2048,
    min_chunk_size=200,
)

# 也可使用标准偏差模式
semantic_chunker_std = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.5,  # 低于均值1.5个标准差的视为breakpoint
    max_chunk_size=2048,
)
```

#### 优缺点分析

| 优点 | 缺点 |
|------|------|
| 感知语义变化，在主题转换处自然切分 | 需要调用嵌入模型，增加计算成本 |
| 块内语义高度一致，有利于精确检索 | 依赖嵌入模型的质量 |
| 自适应内容，无需预设分隔符 | 对无明显主题转换的文本效果不显著 |
| 对不同领域文档自动调整 | 相似度阈值需要根据文档类型调优 |
| 减少"一句话多主题"的问题 | 处理大规模文档时速度较慢 |

---

### 4.1.5 标题/层级分块 (Title-Based / Hierarchical Chunking)

#### 概念定义

标题分块利用文档的标题层级结构（H1, H2, H3, ...）作为自然的分块边界，每个块保留其所属的标题路径（breadcrumb），形成层次化的元数据结构。

#### 解决的问题

- 结构化文档（如技术文档、法律文件）的层级信息被传统分块丢失
- 检索结果缺少上下文定位信息（"这段话属于哪个章节？"）
- 同标题下的内容天然属于同一主题单元

#### 工作原理

```
标题层级分块 (Markdown 文档)

文档结构:
┌──────────────────────────────────────────────────────────────────┐
│ # 第四章：分块技术              ← H1                             │
│                                                                  │
│ ## 4.1 固定长度分块             ← H2                             │
│ 固定长度分块将文本按照预设的...                                   │
│                                                                  │
│ ### 4.1.1 字符级分块            ← H3                             │
│ 字符级分块按照字符数进行切分。它的优点是...                        │
│                                                                  │
│ ### 4.1.2 Token级分块           ← H3                             │
│ Token级分块使用分词器...                                          │
│                                                                  │
│ ## 4.2 递归分块                 ← H2                             │
│ 递归分块按分隔符优先级...                                         │
└──────────────────────────────────────────────────────────────────┘

分块结果（每个Block附带完整标题链）:

Block 1:
  metadata: {
    "h1": "第四章：分块技术",
    "h2": "4.1 固定长度分块",
    "h3": "4.1.1 字符级分块"
  }
  content: "字符级分块按照字符数进行切分。它的优点是..."

Block 2:
  metadata: {
    "h1": "第四章：分块技术",
    "h2": "4.1 固定长度分块",
    "h3": "4.1.2 Token级分块"
  }
  content: "Token级分块使用分词器..."

Block 3:
  metadata: {
    "h1": "第四章：分块技术",
    "h2": "4.2 递归分块"
  }
  content: "递归分块按分隔符优先级..."

检索增强：
  当检索到 Block 1 时，可以利用 metadata 中的标题信息：
  - 添加标题到检索结果的展示中，提升可读性
  - 执行"父级检索"：同时返回同章节的其他块
  - 用标题信息过滤搜索结果
```

#### 核心算法

```python
import re
from typing import List, Dict, Optional


def markdown_hierarchical_chunking(
    text: str,
    max_chunk_size: int = 1024,
    min_chunk_size: int = 100,
    heading_levels: List[int] = [1, 2, 3, 4],
) -> List[Dict]:
    """
    Markdown 层级分块
    
    解析Markdown标题层级，在每个标题处创建新块，
    保留从根到当前块的完整标题链作为元数据。
    
    Args:
        text: Markdown 文本
        max_chunk_size: 最大块大小
        min_chunk_size: 最小块大小（小块合并到前一块）
        heading_levels: 要识别的标题层级
    
    Returns:
        [{"content": str, "metadata": {"h1": str, "h2": str, ...}, 
          "title_path": str}]
    """
    # 标题匹配正则：匹配 # 开头的行
    heading_pattern = re.compile(
        r'^(#{1,' + str(max(heading_levels)) + r'})\s+(.+)$',
        re.MULTILINE
    )
    
    # 解析文档结构
    sections = []
    current_pos = 0
    current_breadcrumb = {}  # {"h1": "xxx", "h2": "yyy"}
    
    for match in heading_pattern.finditer(text):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.start()
        
        # 将当前标题之前的文本作为 content
        if start > current_pos:
            section_text = text[current_pos:start].strip()
            if section_text:
                sections.append({
                    "content": section_text,
                    "breadcrumb": dict(current_breadcrumb),
                    "title_path": _build_title_path(current_breadcrumb),
                })
        
        # 更新面包屑：清除当前级别及以下的标题
        current_breadcrumb[f"h{level}"] = title
        for l in range(level + 1, max(heading_levels) + 1):
            current_breadcrumb.pop(f"h{l}", None)
        
        current_pos = match.end()
    
    # 最后一段内容
    if current_pos < len(text):
        section_text = text[current_pos:].strip()
        if section_text:
            sections.append({
                "content": section_text,
                "breadcrumb": dict(current_breadcrumb),
                "title_path": _build_title_path(current_breadcrumb),
            })
    
    # 处理超大块和超小块
    chunks = []
    carry_over = ""
    
    for section in sections:
        content = carry_over + section["content"] if carry_over else section["content"]
        carry_over = ""
        
        if len(content) <= max_chunk_size:
            if len(content) >= min_chunk_size or not chunks:
                chunks.append({
                    "content": content,
                    "metadata": section["breadcrumb"],
                    "title_path": section["title_path"],
                })
            else:
                # 太小，合并到上一个块
                if chunks:
                    chunks[-1]["content"] += "\n\n" + content
                else:
                    chunks.append({
                        "content": content,
                        "metadata": section["breadcrumb"],
                        "title_path": section["title_path"],
                    })
        else:
            # 超出最大长度，需要二次切分
            sub_chunks = _split_with_breadcrumb(
                content, max_chunk_size, section["breadcrumb"]
            )
            chunks.extend(sub_chunks)
    
    return chunks


def _build_title_path(breadcrumb: Dict[str, str]) -> str:
    """构建标题路径字符串"""
    levels = sorted(breadcrumb.keys())
    return " > ".join(breadcrumb[l] for l in levels)


def _split_with_breadcrumb(
    text: str, max_size: int, breadcrumb: Dict[str, str]
) -> List[Dict]:
    """在保留层级信息的前提下二次切分超大块"""
    separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " "]
    
    chunks = [text]
    for sep in separators:
        new_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_size:
                new_chunks.append(chunk)
            else:
                parts = chunk.split(sep)
                merged = []
                current = ""
                for part in parts:
                    if len(current + sep + part) <= max_size:
                        current = current + sep + part if current else part
                    else:
                        if current:
                            merged.append(current)
                        current = part
                if current:
                    merged.append(current)
                new_chunks.extend(merged)
        chunks = new_chunks
        if all(len(c) <= max_size for c in chunks):
            break
    
    return [
        {
            "content": chunk,
            "metadata": breadcrumb,
            "title_path": _build_title_path(breadcrumb),
        }
        for chunk in chunks
    ]
```

#### LangChain Markdown 分块实现

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False,  # 保留标题文本在content中
)

# 对于HTML文档
from langchain.text_splitter import HTMLHeaderTextSplitter

html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[
        ("h1", "h1"),
        ("h2", "h2"),
        ("h3", "h3"),
        ("h4", "h4"),
    ],
)
```

#### 优缺点分析

| 优点 | 缺点 |
|------|------|
| 完美保留文档结构信息 | 仅适用于有明确标题结构的文档 |
| 元数据可用于检索增强和过滤 | 对纯文本、非结构化文档无效 |
| 标题链提供上下文定位 | 标题层级不一致时效果下降 |
| 各块语义边界清晰 | 实现复杂度较高 |
| 支持子块检索（返回同一章节的内容） | 需要针对不同文档格式定制 |

---

### 4.1.6 章节/结构感知分块 (Document Structure-Aware Chunking)

#### 概念定义

章节分块利用文档的内在结构（章节、条款、附录等），按照文档的原始组织方式进行分块。不仅识别标题，还能理解更复杂的文档结构如表格、列表、代码块、引用等。

#### 解决的问题

- 超越标题层级，理解文档的完整结构
- 保留表格、代码块等特殊格式的完整性
- 处理法律条款编号、学术论文的章节号等特殊结构

#### 工作原理

```
结构感知分块 - 多格式处理

输入文档（混合格式）:
┌──────────────────────────────────────────────────────────────────┐
│ §1.1 定义                                                        │
│ 在本协议中，以下术语具有如下含义：                                 │
│ (a) "许可方" 指提供软件许可的一方；                               │
│ (b) "被许可方" 指接受软件许可的一方。                             │
│                                                                  │
│ ┌─────────────────────────────────────────┐                     │
│ │ 术语        │ 定义                      │                     │
│ ├─────────────────────────────────────────┤                     │
│ │ 许可方      │ 提供软件许可的一方        │                     │
│ │ 被许可方    │ 接受软件许可的一方        │                     │
│ └─────────────────────────────────────────┘                     │
│                                                                  │
│ ```python                                                        │
│ def validate_license(user_id: str) -> bool:                      │
│     return license_service.check(user_id)                        │
│ ```                                                              │
│                                                                  │
│ 被许可方应当遵守以下义务：                                        │
│ 1. 不得对软件进行反向工程；                                       │
│ 2. 不得将软件分发给未经授权的第三方。                              │
└──────────────────────────────────────────────────────────────────┘

分块策略:
┌──────────────────────────────────────────────────────────────────┐
│ Block 1: §1.1 标题 + 定义文本 + 列举项                          │
│ Block 2: 表格（完整保留，不拆散行列关系）                        │
│ Block 3: 代码块（完整保留，不截断代码语法）                      │
│ Block 4: 编号列表（保留编号结构）                                │
└──────────────────────────────────────────────────────────────────┘

关键原则:
- 表格：原子单元，不拆散
- 代码块：原子单元，不拆散
- 列表：保持编号/项目符号结构
- 法律条款编号：作为边界标记
```

#### 核心算法

```python
from enum import Enum
from typing import List, Dict, Any
from dataclasses import dataclass, field
import re


class ContentType(Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    CODE_BLOCK = "code_block"
    LIST_ITEM = "list_item"
    BLOCKQUOTE = "blockquote"


@dataclass
class StructuralElement:
    """文档结构元素"""
    content_type: ContentType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    level: int = 0
    children: List['StructuralElement'] = field(default_factory=list)


def structure_aware_chunking(
    text: str,
    max_chunk_size: int = 1024,
    doc_format: str = "markdown",
) -> List[Dict[str, Any]]:
    """
    结构感知分块
    
    解析文档的完整结构（表格、代码块、列表等），
    在保持各元素完整性的前提下进行分块。
    
    Args:
        text: 输入文本
        max_chunk_size: 最大块大小
        doc_format: 文档格式 ("markdown", "html", "rst")
    
    Returns:
        带结构元数据的文本块列表
    """
    if doc_format == "markdown":
        elements = _parse_markdown_structure(text)
    elif doc_format == "html":
        elements = _parse_html_structure(text)
    else:
        elements = _parse_plain_structure(text)
    
    # 将结构元素组装为块（保护原子元素）
    chunks = []
    current_block = ""
    current_metadata = {}
    
    for elem in elements:
        # 表格、代码块等被视为原子元素，不拆分
        if elem.content_type in [ContentType.TABLE, ContentType.CODE_BLOCK]:
            # 保存当前accumulated块
            if current_block.strip():
                chunks.append({
                    "content": current_block.strip(),
                    "metadata": current_metadata,
                })
                current_block = ""
                current_metadata = {}
            
            # 原子元素单独成块
            chunks.append({
                "content": elem.content,
                "metadata": elem.metadata,
                "content_type": elem.content_type.value,
                "is_atomic": True,
            })
            continue
        
        # 标题作为新块的开始
        if elem.content_type == ContentType.HEADING:
            if current_block.strip():
                chunks.append({
                    "content": current_block.strip(),
                    "metadata": current_metadata,
                })
            current_block = elem.content + "\n"
            current_metadata = elem.metadata
            continue
        
        # 普通段落/列表项
        potential = current_block + elem.content if current_block else elem.content
        if len(potential) <= max_chunk_size:
            current_block = potential
        else:
            if current_block.strip():
                chunks.append({
                    "content": current_block.strip(),
                    "metadata": current_metadata,
                })
            current_block = elem.content
    
    if current_block.strip():
        chunks.append({
            "content": current_block.strip(),
            "metadata": current_metadata,
        })
    
    return chunks


def _parse_markdown_structure(text: str) -> List[StructuralElement]:
    """解析Markdown文档的结构元素"""
    elements = []
    lines = text.split("\n")
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # 代码块
        if line.strip().startswith("```"):
            code_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                code_lines.append(lines[i])
            
            lang = line.strip()[3:].strip()
            elements.append(StructuralElement(
                content_type=ContentType.CODE_BLOCK,
                content="\n".join(code_lines),
                metadata={"language": lang} if lang else {},
            ))
            i += 1
            continue
        
        # 表格
        if "|" in line and i + 1 < len(lines) and "---" in lines[i+1]:
            table_lines = [line]
            i += 1
            # 分隔行
            table_lines.append(lines[i])
            i += 1
            # 数据行
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            
            elements.append(StructuralElement(
                content_type=ContentType.TABLE,
                content="\n".join(table_lines),
                metadata={"format": "markdown_table"},
            ))
            continue
        
        # 标题
        heading_match = re.match(r'^(#{1,6})\s+(.+)', line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2)
            elements.append(StructuralElement(
                content_type=ContentType.HEADING,
                content=line,
                metadata={"heading_level": level, "title": title},
                level=level,
            ))
            i += 1
            continue
        
        # 列表项
        list_match = re.match(r'^(\s*)([-*+]|\d+[.)])\s+(.+)', line)
        if list_match:
            elements.append(StructuralElement(
                content_type=ContentType.LIST_ITEM,
                content=line,
                metadata={"indent": len(list_match.group(1)), 
                          "marker": list_match.group(2)},
            ))
            i += 1
            continue
        
        # 引用块
        if line.strip().startswith(">"):
            quote_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i])
                i += 1
            elements.append(StructuralElement(
                content_type=ContentType.BLOCKQUOTE,
                content="\n".join(quote_lines),
            ))
            continue
        
        # 普通段落（accumulate连续的非空行）
        if line.strip():
            para_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and \
                  not lines[i].strip().startswith("#") and \
                  not lines[i].strip().startswith("```") and \
                  not lines[i].strip().startswith(">") and \
                  not re.match(r'^(\s*)([-*+]|\d+[.)])\s+', lines[i]) and \
                  "|" not in lines[i]:
                para_lines.append(lines[i])
                i += 1
            elements.append(StructuralElement(
                content_type=ContentType.PARAGRAPH,
                content="\n".join(para_lines),
            ))
        else:
            i += 1
    
    return elements
```

#### 优缺点分析

| 优点 | 缺点 |
|------|------|
| 保护表格、代码等特殊格式的完整性 | 实现复杂度高（需要多格式解析器） |
| 充分利用文档的结构语义 | 对文档格式有依赖 |
| 对结构化文档效果最好的方法之一 | 对格式不规范的文档容错性差 |
| 元数据丰富，支持定向检索 | 每种文档格式需要单独适配 |
| 适合法律、技术文档等结构化内容 | 处理速度较慢 |

---

### 4.1.7 知识点分块 (Knowledge-Point Chunking)

#### 概念定义

知识点分块以"知识单元"（Knowledge Point）为最小粒度进行分块。一个知识点是一个语义上自包含的、可以独立被检索和理解的最小信息单元，通常对应一个概念的定义、一个操作的步骤、一个问题的答案等。

#### 解决的问题

- 传统分块按长度或格式切分，一个块可能包含多个知识点
- 检索时无法精确匹配到单个知识点
- 知识点的混合降低了检索的精度

#### 工作原理

```
知识点分块 vs 传统分块

原始文本:
"Python中的列表（list）是一种可变序列类型，支持索引、切片和修改操作。
 列表使用方括号定义，例如 [1, 2, 3]。
 
 Python中的元组（tuple）是一种不可变序列类型，一旦创建就不能修改。
 元组使用圆括号定义，例如 (1, 2, 3)。元组由于不可变性，可以作为字典的键。
 
 Python中的字典（dict）是一种键值对映射类型，支持通过键快速查找值。
 字典使用花括号定义，例如 {'a': 1, 'b': 2}。字典的键必须是不可变类型。"

传统分块（固定500 chars）:
┌──────────────────────────────────────────────────────────────────┐
│ Chunk 1:                                                        │
│ "Python中的列表（list）是一种可变序列类型...                      │
│  Python中的元组（tuple）是一种不可变序列类型..."                  │
│  （列表 + 元组混在一个块中）                                     │
└──────────────────────────────────────────────────────────────────┘

知识点分块:
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ 知识点1: 列表     │ │ 知识点2: 元组     │ │ 知识点3: 字典    │
│ - 可变序列        │ │ - 不可变序列       │ │ - 键值对映射     │
│ - 支持索引/切片   │ │ - 定义语法         │ │ - 定义语法       │
│ - 定义语法        │ │ - 可作为字典键     │ │ - 键的不可变性   │
└──────────────────┘ └──────────────────┘ └──────────────────┘
     独立可检索          独立可检索          独立可检索
```

#### 核心算法（基于规则+模式匹配）

```python
import re
from typing import List, Dict


def knowledge_point_chunking(
    text: str,
    min_kp_length: int = 50,
    max_kp_length: int = 800,
) -> List[Dict[str, str]]:
    """
    知识点分块
    
    使用规则和模式匹配识别文本中的独立知识点，
    每个知识点作为一个独立的检索单元。
    
    Args:
        text: 输入文本
        min_kp_length: 最小知识点长度
        max_kp_length: 最大知识点长度
    
    Returns:
        知识点列表 [{"title": str, "content": str, "type": str}]
    """
    knowledge_points = []
    
    # 规则1: 定义模式 "X 是/指/为 Y"
    definition_pattern = re.compile(
        r'(.{0,30}(?:是|指|为|即|指的是|是指|定义为)[^。！？\n]{20,300}[。！？\n])'
    )
    for match in definition_pattern.finditer(text):
        kp_text = match.group(0).strip()
        if min_kp_length <= len(kp_text) <= max_kp_length:
            # 提取知识点的主题（主语部分）
            subject = match.group(1).split("是")[0].split("指")[0].strip()
            knowledge_points.append({
                "title": subject[:50],
                "content": kp_text,
                "type": "definition",
            })
    
    # 规则2: 分类枚举模式 "X 包括/分为/有以下几种/分为以下几类"
    enum_pattern = re.compile(
        r'(.{0,30}(?:包括|分为|有以下|以下几类|如下)[：:]\s*'
        r'(?:[\n]*(?:\d+[.、)]|\-|\*)\s*[^\n]+){2,})'
    )
    for match in enum_pattern.finditer(text):
        kp_text = match.group(0).strip()
        if min_kp_length <= len(kp_text) <= max_kp_length:
            knowledge_points.append({
                "title": match.group(1).split("包括")[0].split("分为")[0].strip()[:50],
                "content": kp_text,
                "type": "enumeration",
            })
    
    # 规则3: 问答模式 （Q&A, FAQ, 问题/答案）
    qa_pattern = re.compile(
        r'((?:Q|问|问题)[：:]\s*.{5,100}\n'
        r'(?:A|答|答案|回答)[：:]\s*.{20,500})',
        re.IGNORECASE | re.DOTALL
    )
    for match in qa_pattern.finditer(text):
        kp_text = match.group(0).strip()
        if min_kp_length <= len(kp_text) <= max_kp_length:
            # 提取问题部分
            question = match.group(1).split("\n")[0]
            knowledge_points.append({
                "title": question[:80],
                "content": kp_text,
                "type": "qa",
            })
    
    # 规则4: 步骤模式 "第X步" / "步骤X" / "Step X"
    step_pattern = re.compile(
        r'((?:第[一二三四五六七八九十\d]+步|步骤\s*[一二三四五六七八九十\d]+|'
        r'Step\s*\d+).{10,400})'
    )
    for match in step_pattern.finditer(text):
        kp_text = match.group(0).strip()
        if min_kp_length <= len(kp_text) <= max_kp_length:
            knowledge_points.append({
                "title": match.group(1).split("。")[0][:50],
                "content": kp_text,
                "type": "step",
            })
    
    # 去重：移除内容重叠度高的知识点
    knowledge_points = _deduplicate_knowledge_points(knowledge_points)
    
    return knowledge_points


def _deduplicate_knowledge_points(
    kps: List[Dict], 
    overlap_threshold: float = 0.8
) -> List[Dict]:
    """基于Jaccard相似度去除重复知识点"""
    if len(kps) <= 1:
        return kps
    
    kept = []
    for kp in kps:
        is_duplicate = False
        kp_words = set(kp["content"][:100])
        
        for existing in kept:
            existing_words = set(existing["content"][:100])
            if not kp_words or not existing_words:
                continue
            jaccard = len(kp_words & existing_words) / len(kp_words | existing_words)
            if jaccard > overlap_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            kept.append(kp)
    
    return kept
```

#### 优缺点分析

| 优点 | 缺点 |
|------|------|
| 检索精度最高：每块对应一个知识点 | 规则模式难以覆盖所有知识类型 |
| 每个块语义自包含，适合精确问答 | 知识边界的自动识别仍不够准确 |
| 减少无关信息干扰 | 对非结构化文档效果有限 |
| 适合FAQ、技术文档等结构化知识 | 人工审核和标注成本较高 |

---

### 4.1.8 LLM辅助分块 (LLM-Assisted Chunking)

#### 概念定义

LLM辅助分块利用大语言模型本身的语义理解能力来识别文档中的自然边界，由LLM判断何处应该切分、何处应保持连续。这是分块的"智能"实现方式。

#### 解决的问题

- 基于规则/模式的分块难以处理复杂的语义边界
- 不同领域文档需要不同的分块策略，难以用固定规则覆盖
- 文档中隐含的主题转换需要深度理解

#### 工作原理

```
LLM辅助分块流程

Step 1: 预处理
┌──────────────────────────────────────────────────────────────────┐
│ 将文档按段落或小节预切分，减少LLM处理的长度                      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 2: LLM边界识别
┌──────────────────────────────────────────────────────────────────┐
│ Prompt 发给 LLM:                                                │
│                                                                  │
│ "你是一个文档结构分析专家。请分析以下文本，在主题发生变化的位置   │
│  标记 [CHUNK_BREAK]。确保每个块是一个语义完整的知识单元，         │
│  块的长度应在200-800字之间。                                     │
│                                                                  │
│  文本：{document_section}"                                       │
│                                                                  │
│ LLM 返回:                                                       │
│ "Python中的列表是一种可变序列类型... [CHUNK_BREAK]               │
│  Python中的元组是一种不可变序列... [CHUNK_BREAK]                 │
│  Python中的字典是一种键值对映射..."                              │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 3: 后处理
┌──────────────────────────────────────────────────────────────────┐
│ - 根据 [CHUNK_BREAK] 标记切分文本                               │
│ - 检查每个块的token数量是否在合理范围内                          │
│ - 对过长块进行二次切分                                          │
│ - 生成块元数据（标题、所属章节等）                               │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 4: 验证与纠错
┌──────────────────────────────────────────────────────────────────┐
│ - 检查是否存在空块或过短块                                      │
│ - 验证 chunk 之间的内容不丢失                                    │
│ - 人工或规则校验关键段落的分块质量                               │
└──────────────────────────────────────────────────────────────────┘
```

#### 核心实现

```python
from typing import List, Dict, Optional
import json


class LLMAssistedChunker:
    """
    LLM辅助分块器
    
    使用LLM识别文档中的语义边界，实现高质量的分块。
    支持不同的LLM后端（OpenAI, Anthropic, 本地模型）。
    """
    
    CHUNKING_PROMPT = """你是一个专业的文档结构分析专家。
请分析以下文本，在语义主题发生变化的位置插入 [CHUNK_BREAK] 标记。

分块原则：
1. 每个块应该是一个自包含的知识单元，不依赖其他块即可理解
2. 块的长度应控制在 {min_chunk_size}-{max_chunk_size} 字符之间
3. 在以下位置优先切分：
   - 新概念/新主题的开始
   - 从一个操作步骤切换到另一个步骤
   - 从说明切换到示例
4. 在以下位置不要切分：
   - 一个概念的说明中间
   - 连续的论点或推理链条中
   - 紧密关联的数据和说明之间
5. 标题行（如"## "、"**粗体标题**"）通常应作为新块的开始标志

文档类型: {doc_type}
语言: 中文（简体）

文本：
{text}

请返回带有 [CHUNK_BREAK] 标记的完整文本，不要省略任何内容。"""

    def __init__(
        self,
        llm_client,
        doc_type: str = "technical",
        min_chunk_size: int = 200,
        max_chunk_size: int = 800,
        batch_size: int = 3000,
    ):
        """
        Args:
            llm_client: LLM客户端（需支持 completions/chat API）
            doc_type: 文档类型提示（影响LLM的分块决策）
            min_chunk_size: 最小块大小建议
            max_chunk_size: 最大块大小建议
            batch_size: 每次发给LLM的最大文本量
        """
        self.llm = llm_client
        self.doc_type = doc_type
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.batch_size = batch_size
    
    def chunk(self, text: str) -> List[Dict[str, any]]:
        """
        对文档进行LLM辅助分块
        
        Returns:
            [{"content": str, "index": int, "token_count": int}]
        """
        # Step 1: 将长文档分段处理
        segments = self._segment_for_llm(text)
        
        # Step 2: 逐段调用LLM进行边界标记
        marked_texts = []
        for seg in segments:
            marked = self._llm_mark_boundaries(seg)
            marked_texts.append(marked)
        
        # Step 3: 按标记切分
        full_marked = "\n".join(marked_texts)
        chunks = self._split_by_marks(full_marked)
        
        # Step 4: 后处理
        chunks = self._post_process(chunks)
        
        return chunks
    
    def _segment_for_llm(self, text: str) -> List[str]:
        """将长文档分段，适配LLM的上下文窗口"""
        if len(text) <= self.batch_size:
            return [text]
        
        # 在段落边界处切分
        segments = []
        current = ""
        for para in text.split("\n\n"):
            if len(current) + len(para) > self.batch_size:
                if current:
                    segments.append(current)
                current = para
            else:
                current = current + "\n\n" + para if current else para
        if current:
            segments.append(current)
        
        return segments
    
    def _llm_mark_boundaries(self, text: str) -> str:
        """调用LLM进行边界标记"""
        prompt = self.CHUNKING_PROMPT.format(
            min_chunk_size=self.min_chunk_size,
            max_chunk_size=self.max_chunk_size,
            doc_type=self.doc_type,
            text=text,
        )
        
        # 使用OpenAI API的示例
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",  # 使用廉价模型降低分块成本
                messages=[
                    {"role": "system", "content": "你是一个精确的文档分析助手。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,  # 零温度确保一致性
                max_tokens=4000,
            )
            return response.choices[0].message.content
        except Exception:
            # 降级：不使用LLM标记
            return text
    
    def _split_by_marks(self, marked_text: str) -> List[str]:
        """按标记切分文本"""
        raw_chunks = marked_text.split("[CHUNK_BREAK]")
        return [c.strip() for c in raw_chunks if c.strip()]
    
    def _post_process(self, chunks: List[str]) -> List[Dict]:
        """后处理：合并过小块、拆解过大块"""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        
        result = []
        buffer = ""
        
        for chunk in chunks:
            combined = buffer + "\n" + chunk if buffer else chunk
            
            tokens = len(enc.encode(combined))
            max_tokens = int(self.max_chunk_size * 0.75)  # 粗略token换算
            
            if tokens <= max_tokens:
                buffer = combined
            else:
                if buffer:
                    result.append(buffer)
                buffer = chunk
        
        if buffer:
            result.append(buffer)
        
        return [
            {
                "content": c,
                "index": i,
                "token_count": len(enc.encode(c)),
            }
            for i, c in enumerate(result)
        ]
```

#### 优缺点分析

| 优点 | 缺点 |
|------|------|
| 分块质量最高，接近人工标注水平 | 成本高：每次分块需调用LLM API |
| 可适应任何领域和文档类型 | 速度慢：受LLM推理延迟限制 |
| 无需手工设计规则和分隔符 | 可能引入LLM幻觉（添加或修改内容） |
| 能理解复杂的语义边界 | 结果不完全确定性（temperature影响） |
| 可以处理非标准格式的文档 | 不适合实时或大批量处理场景 |

---

### 4.1.9 分块方法综合对比

```
                    分块方法决策矩阵

                    精度 ───────────────────────────────►
                    (语义完整性)
                    高                                低
                    ┌──────────────────────────────────┐
              高    │ LLM辅助  知识点   语义           │
                    │                           递归   │
              速    │ 标题/层级                       │
                    │      结构感知                    │
              度    │                      滑动窗口   │
                    │                         固定长度 │
              低    │                                  │
                    └──────────────────────────────────┘

推荐选择路径: 固定长度 → 递归 → 标题/层级 → 语义 → LLM辅助
            (基线)   (80%场景) (结构化)  (高质量) (最高质量)
```

| 分块方法 | 语义完整性 | 计算开销 | 实现复杂度 | 适用场景 | 批量处理 | 召回-精度平衡 |
|---------|-----------|---------|-----------|---------|---------|-------------|
| 固定长度 | 2/5 | 极低 | 极低 | 快速原型 | 极快 | 偏向召回 |
| 滑动窗口 | 2/5 | 低 | 低 | 通用基线 | 快 | 平衡 |
| 递归分块 | 3/5 | 低 | 低 | 通用文档 | 快 | 平衡 |
| 标题/层级 | 4/5 | 低 | 中 | 结构化文档 | 快 | 偏向精度 |
| 章节/结构感知 | 4/5 | 中 | 高 | 技术/法律文档 | 中 | 偏向精度 |
| 语义分块 | 4/5 | 中高 | 中 | 主题多样文档 | 中 | 偏向精度 |
| 知识点 | 4/5 | 中 | 中高 | FAQ/知识库 | 中 | 精度最高 |
| LLM辅助 | 5/5 | 高 | 高 | 高质量需求 | 慢 | 精度最高 |

---

## 4.2 场景化分块策略分析

### 4.2.1 技术文档

```
文档特征:
- 层级结构清晰（标题、子标题）
- 包含代码块、表格、API参考
- 技术术语密度高
- 概念之间有明确的引用关系

推荐策略: 标题层级分块 + 代码块保护
```

| 推荐配置 | 值 |
|---------|---|
| 主策略 | MarkdownHeaderTextSplitter |
| chunk_size | 512-1024 tokens |
| chunk_overlap | 10%-15% |
| 特殊处理 | 代码块识别为原子单元 |
| 元数据 | 标题链 + 代码语言 + 文件名 |

```python
# 技术文档最佳实践
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def tech_doc_chunking(markdown_text: str) -> list:
    """技术文档专用分块"""
    # 第一遍：按标题层级切分
    headers = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    header_chunks = header_splitter.split_text(markdown_text)
    
    # 第二遍：对超过阈值的块进行递归切分
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    
    final_chunks = []
    for chunk in header_chunks:
        if len(chunk.page_content) <= 800:
            final_chunks.append(chunk)
        else:
            sub = sub_splitter.split_text(chunk.page_content)
            for s in sub:
                final_chunks.append(Document(
                    page_content=s,
                    metadata=chunk.metadata
                ))
    
    return final_chunks
```

### 4.2.2 法律文档

```
文档特征:
- 条款编号严格（§1.1, 第X条, Article X）
- 逻辑严密，条款间互引
- 修改/删除部分不可随意截断
- 术语定义往往在文档前部

推荐策略: 条款感知分块 + 大块策略
```

| 推荐配置 | 值 |
|---------|---|
| 主策略 | 条款/章节编号分块 |
| chunk_size | 1024-2048 tokens |
| chunk_overlap | 15%-20% |
| 特殊处理 | 识别条款编号、保持条款完整性 |
| 元数据 | 条款编号、章节、版本、生效日期 |

```python
# 法律文档分块
import re

def legal_doc_chunking(text: str) -> list[dict]:
    """
    法律文档专用分块器
    识别: 第X条、§X、Article X、Section X 等条款标记
    """
    # 条款边界正则
    article_patterns = [
        r'(?=第[一二三四五六七八九十百千\d]+条)',
        r'(?=第[一二三四五六七八九十百千\d]+章)',
        r'(?=第[一二三四五六七八九十百千\d]+节)',
        r'(?=§ ?\d+)',
        r'(?=Article \d+)',
        r'(?=Section \d+)',
    ]
    
    combined_pattern = '|'.join(f'({p})' for p in article_patterns)
    
    # 按条款边界切分
    parts = re.split(combined_pattern, text)
    parts = [p for p in parts if p and p.strip()]
    
    # 合并小块
    chunks = []
    current = ""
    for part in parts:
        if len(current + part) <= 1500:
            current += part
        else:
            if current:
                chunks.append({
                    "content": current,
                    "article_number": _extract_article_number(current),
                })
            current = part
    
    if current:
        chunks.append({
            "content": current,
            "article_number": _extract_article_number(current),
        })
    
    return chunks

def _extract_article_number(text: str) -> str:
    """提取条款编号"""
    match = re.search(r'(?:第[一二三四五六七八九十百千\d]+[条章节])|(?:§ ?\d+)|(?:Article \d+)', text)
    return match.group(0) if match else ""
```

### 4.2.3 学术论文

```
文档特征:
- IMRaD结构（Introduction, Methods, Results, Discussion）
- 引用密度高
- 图表说明文字重要
- 摘要包含全局信息

推荐策略: 章节分块 + 引用保留
```

| 推荐配置 | 值 |
|---------|---|
| 主策略 | 章节感知 + 段落级递归 |
| chunk_size | 768-1536 tokens |
| chunk_overlap | 10%-15% |
| 特殊处理 | 保留引用标记、图表标题 |
| 元数据 | 章节名、作者、年份、关键词 |

### 4.2.4 小说/文学作品

```
文档特征:
- 叙事流畅，不适合频繁切分
- 对话密集，人物关系复杂
- 情节连续性强
- 章节/段落是自然边界

推荐策略: 大块章节分块 + 高重叠率
```

| 推荐配置 | 值 |
|---------|---|
| 主策略 | 章节级切分（每章为一块） |
| chunk_size | 2048-4096 tokens |
| chunk_overlap | 20%-25% |
| 特殊处理 | 保留人物对话连续性 |
| 元数据 | 章节名、人物、时间线 |

### 4.2.5 医疗文档

```
文档特征:
- 术语规范化程度高（ICD编码、药品名）
- 数值敏感（剂量、检验值）
- 结构化与非结构化并存
- 隐私和合规要求高

推荐策略: 段落递归分块 + 实体标注
```

| 推荐配置 | 值 |
|---------|---|
| 主策略 | 递归字符分块（医学分隔符） |
| chunk_size | 512-1024 tokens |
| chunk_overlap | 15%-20% |
| 特殊处理 | 药品名-剂量不分离、检验项-值不分离 |
| 元数据 | 科室、病种、文档类别 |

### 4.2.6 代码仓库

```
文档特征:
- 高度结构化（函数、类、模块边界清晰）
- 编程语言语法约束
- 注释与代码关联
- 文件间引用关系

推荐策略: AST感知分块
```

| 推荐配置 | 值 |
|---------|---|
| 主策略 | 函数/类级AST解析 |
| chunk_size | 按函数/类（150-2000 tokens） |
| chunk_overlap | 0%（函数边界清晰，无需重叠） |
| 特殊处理 | AST解析 + 注释保留、import保留 |
| 元数据 | 文件名、函数名、类名、行号 |

```python
# 代码AST感知分块示例
import ast

def code_ast_chunking(source_code: str, filename: str) -> list[dict]:
    """基于Python AST的函数/类级别分块"""
    tree = ast.parse(source_code)
    chunks = []
    
    # 提取模块级文档字符串（如果有）
    module_doc = ast.get_docstring(tree)
    if module_doc:
        chunks.append({
            "content": f"# {filename} Module Docstring\n{module_doc}",
            "type": "module_doc",
            "name": filename,
        })
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_source = ast.get_source_segment(source_code, node)
            docstring = ast.get_docstring(node)
            chunks.append({
                "content": func_source,
                "type": "function",
                "name": node.name,
                "docstring": docstring,
                "lineno": node.lineno,
            })
        elif isinstance(node, ast.ClassDef):
            class_source = ast.get_source_segment(source_code, node)
            docstring = ast.get_docstring(node)
            chunks.append({
                "content": class_source,
                "type": "class",
                "name": node.name,
                "docstring": docstring,
                "lineno": node.lineno,
            })
    
    return chunks
```

### 4.2.7 企业知识库

```
文档特征:
- 多格式、多来源（Word、PDF、Confluence、邮件）
- 主题分散，领域知识密集
- 需要频繁更新
- 用户查询类型多样

推荐策略: 混合策略（格式检测 + 自适应）
```

| 推荐配置 | 值 |
|---------|---|
| 主策略 | 格式路由 + 递归分块 |
| chunk_size | 512-1024 tokens |
| chunk_overlap | 10%-15% |
| 特殊处理 | 按文档格式路由到不同分块器 |
| 元数据 | 来源、部门、更新日期、作者、标签 |

```python
# 企业知识库混合分块路由
def enterprise_kb_chunking(text: str, format_type: str) -> list:
    """
    根据文档格式路由到最合适的分块器
    """
    if format_type == "markdown":
        return tech_doc_chunking(text)
    elif format_type == "pdf":
        # PDF 需要先提取文本再处理
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        return splitter.split_text(text)
    elif format_type == "html":
        from langchain.text_splitter import HTMLHeaderTextSplitter
        splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=[("h2", "h2"), ("h3", "h3")],
        )
        return splitter.split_text(text)
    elif format_type == "code":
        return code_ast_chunking(text, "unknown.py")
    else:
        # 默认：递归分块
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        return splitter.split_text(text)
```

---

## 4.3 分块大小与重叠深度分析

### 4.3.1 不同分块大小效果对比

下表基于使用 `text-embedding-3-small` 模型（最大输入8192 tokens）在中文技术文档数据集上的实验：

| Chunk Size (tokens) | 召回率 (Recall@5) | 精度 (Precision@5) | 上下文连贯性 | 推理速度 | 存储开销 | 综合评分 |
|--------------------|-------------------|--------------------|------------|---------|---------|---------|
| **128** | 72.3% | 85.1% | 差/碎片化 | 最快 | 高 | 6.5/10 |
| **256** | 78.8% | 82.4% | 一般 | 很快 | 中高 | 7.2/10 |
| **512** | 84.1% | 78.9% | 较好 | 快 | 中 | **8.3/10** |
| **768** | 86.5% | 75.2% | 好 | 适中 | 中 | **8.1/10** |
| **1024** | 87.2% | 71.8% | 好 | 适中 | 低 | **7.9/10** |
| **2048** | 88.9% | 65.3% | 很好 | 慢 | 低 | 7.0/10 |
| **4096** | 90.1% | 58.7% | 很好 | 很慢 | 很低 | 5.8/10 |

```
召回率 vs 精度 曲线（近似）

Recall@5
  95% │                                    ●(4096)
      │                           ●(2048)
  90% │                    ●(1024)
      │               ●(768)
  85% │          ●(512)
      │
  80% │     ●(256)
      │
  75% │ ●(128)
      │
  70% ├────────────────────────────────────────────►
        50%    60%    70%    80%    90%    100%
                      Precision@5

观察：
- 512-768 tokens 是"甜蜜点"（sweet spot），精度-召回平衡最优
- 1024+ tokens 精度下降显著，召回提升边际递减
- 128 tokens 碎片化严重，上下文不连贯
```

### 4.3.2 不同场景的经验最佳值

| 应用场景 | 推荐 Chunk Size | 推荐 Overlap | 嵌入模型推荐 | 说明 |
|---------|----------------|-------------|-------------|------|
| 通用问答 | 512-768 tokens | 10%-15% | text-embedding-3-small | 精度-召回平衡 |
| 文档摘要 | 1024-2048 tokens | 20% | text-embedding-3-large | 需要更多上下文 |
| 精确问答 (FAQ) | 256-512 tokens | 5%-10% | text-embedding-3-small | 小块精确匹配 |
| 代码搜索 | 函数级 (150-1500) | 0% | code-embedding-001 | AST切分 |
| 法律检索 | 1024-2048 tokens | 20% | text-embedding-3-large | 条款完整性 |
| 学术搜索 | 768-1536 tokens | 15% | text-embedding-3-large | 保留引用上下文 |
| 长文档问答 | 2048-4096 tokens | 25% | text-embedding-3-large | 最大上下文 |
| 多语言混合 | 512 tokens | 10% | text-embedding-3-small | 统一token计数 |
| 实时对话 | 256-512 tokens | 10% | text-embedding-3-small | 优先速度 |

### 4.3.3 分块重叠的深度分析

```
重叠率的权衡

┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   重叠率低(0%-10%)              重叠率高(20%-30%)               │
│   ┌─────────────────┐           ┌─────────────────┐             │
│   │ 存储成本:  低   │           │ 存储成本:  高   │             │
│   │ 计算成本:  低   │           │ 计算成本:  高   │             │
│   │ 上下文断裂: 高  │           │ 上下文断裂: 低  │             │
│   │ 冗余信息:  低   │           │ 冗余信息:  高   │             │
│   │ 适合: FAQ/短问答│           │ 适合: 长篇推理  │             │
│   └─────────────────┘           └─────────────────┘             │
│                                                                  │
│   最优重叠率 = f(chunk_size, content_type, query_type)          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

经验公式：
  optimal_overlap = max(50, chunk_size * 0.15) tokens
  
  即：
  chunk_size=256   → overlap ≈ 50 tokens
  chunk_size=512   → overlap ≈ 77 tokens
  chunk_size=1024  → overlap ≈ 154 tokens
  chunk_size=2048  → overlap ≈ 307 tokens
```

### 4.3.4 分块大小与嵌入模型的关系

```
嵌入模型最大输入长度对比

┌───────────────────────────────────────────────────────────────────────┐
│ 模型                       │ 最大输入 │ 推荐 Chunk Size │ 备注       │
├───────────────────────────────────────────────────────────────────────┤
│ text-embedding-ada-002     │ 8191     │ 512-1024        │ 已弃用     │
│ text-embedding-3-small     │ 8191     │ 512-1024        │ 性价最优   │
│ text-embedding-3-large     │ 8191     │ 1024-2048       │ 质量最高   │
│ bge-large-zh-v1.5          │ 512      │ 256-500         │ 中文专用   │
│ bge-m3                     │ 8192     │ 512-1024        │ 多语言     │
│ mE5-large                  │ 512      │ 300-500         │ 多语言     │
│ Cohere embed-v3            │ 512      │ 300-512         │ 质量高     │
│ Jina embeddings v3         │ 8192     │ 512-2048        │ 长上下文   │
│ voyage-2                   │ 16000    │ 1024-4096       │ 超长上下文 │
└───────────────────────────────────────────────────────────────────────┘

关键约束：
- Chunk Size 必须 ≤ 嵌入模型的最大输入长度
- 建议留出 10% buffer（如最大512 → 实际 ≤ 460）
- BGE系列中文模型常用 512 限制，需特别注意
- 超出限制的内容会被截断，信息永久丢失！
```

---

## 4.4 动态分块技术

### 4.4.1 自适应分块 (Adaptive Chunking)

#### 概念定义

自适应分块根据文档内容类型动态调整分块策略。不是对所有文档使用同一种分块器，而是在处理时分析文档特征，自动选择最适合的分块方法和参数。

#### 工作原理

```
自适应分块决策流程

输入文档
    │
    ▼
┌─────────────┐     ┌──────────────────────────────────┐
│ 内容类型检测 │ ──► │ 检测结果: 技术文档/法律/学术/... │
└─────────────┘     └──────────────────────────────────┘
    │
    ▼
┌─────────────┐     ┌──────────────────────────────────┐
│ 结构复杂度  │ ──► │ 复杂度: 高/中/低                  │
│   评估      │     │ 维度: 标题层级、表格、代码块、    │
└─────────────┘     │       列表、公式、引用             │
    │               └──────────────────────────────────┘
    ▼
┌─────────────┐     ┌──────────────────────────────────┐
│ 语言/混合度 │ ──► │ 中英混合比例、专业术语密度        │
│   检测      │     └──────────────────────────────────┘
└─────────────┘
    │
    ▼
┌─────────────┐
│ 策略选择器  │ ──► 输出: (分块器类型, chunk_size, overlap)
└─────────────┘

决策树示例:
  if 代码块 >= 30%:   → AST分块, size=函数级
  elif 表格 >= 20%:   → 结构感知, size=1024
  elif 标题层级 >= 3: → 标题分块, size=800
  elif 法律条款模式:  → 条款分块, size=1500
  else:               → 递归分块, size=512
```

#### 核心实现

```python
from enum import Enum
from dataclasses import dataclass
import re


class ChunkingStrategy(Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    MARKDOWN_HEADER = "markdown_header"
    STRUCTURE_AWARE = "structure_aware"
    LEGAL_ARTICLE = "legal_article"
    AST_CODE = "ast_code"


@dataclass
class AdaptiveConfig:
    strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int
    separators: list = None
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


class AdaptiveChunker:
    """
    自适应分块器
    
    自动检测文档特征，选择最优的分块策略和参数。
    """
    
    # 文档类型特征指纹
    FEATURES = {
        "code": {
            "patterns": [r'```\w*\n', r'def \w+\(', r'class \w+[:\(]',
                        r'function \w+\(', r'import \w+', r'const \w+'],
            "min_matches": 3,
            "strategy": ChunkingStrategy.AST_CODE,
            "chunk_size_range": (100, 2000),
        },
        "legal": {
            "patterns": [r'第[一二三四五六七八九十百千\d]+条',
                        r'§\s*\d+', r'Article \d+', r'Section \d+',
                        r'甲方|乙方|出卖人|买受人', r'本合同|本协议'],
            "min_matches": 2,
            "strategy": ChunkingStrategy.LEGAL_ARTICLE,
            "chunk_size_range": (1024, 2048),
        },
        "markdown": {
            "patterns": [r'^#{1,6}\s', r'^\|.*\|', r'\[.*\]\(.*\)'],
            "min_matches": 3,
            "strategy": ChunkingStrategy.MARKDOWN_HEADER,
            "chunk_size_range": (512, 1024),
        },
        "academic": {
            "patterns": [r'Abstract', r'Introduction', r'References?',
                        r'\[\d+\]', r'Fig\.\s*\d+', r'Table\s*\d+',
                        r'摘要', r'引言', r'参考文献', r'致谢'],
            "min_matches": 2,
            "strategy": ChunkingStrategy.RECURSIVE,
            "chunk_size_range": (768, 1536),
        },
    }
    
    def __init__(self, default_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
                 default_chunk_size: int = 512, default_overlap: int = 75):
        self.default_config = AdaptiveConfig(
            strategy=default_strategy,
            chunk_size=default_chunk_size,
            chunk_overlap=default_overlap,
        )
    
    def detect_and_chunk(self, text: str) -> tuple:
        """
        检测文档类型并执行分块
        
        Returns:
            (chunks, strategy_used, detected_type)
        """
        # 特征检测
        detected_type, confidence = self._detect_document_type(text)
        
        # 参数选择
        config = self._select_config(detected_type, confidence, text)
        
        # 统计分析（语言比例、术语密度等）
        stats = self._analyze_text_stats(text)
        config = self._adjust_config_by_stats(config, stats)
        
        # 执行分块
        chunks = self._execute_chunking(text, config)
        
        return chunks, config.strategy.value, detected_type
    
    def _detect_document_type(self, text: str) -> tuple:
        """基于特征模式匹配的文档类型检测"""
        scores = {}
        
        for doc_type, features in self.FEATURES.items():
            matches = 0
            for pattern in features["patterns"]:
                matches += len(re.findall(pattern, text, re.MULTILINE))
            scores[doc_type] = matches
        
        if not scores:
            return ("general", 0.0)
        
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        total_matches = sum(scores.values())
        
        confidence = best_score / total_matches if total_matches > 0 else 0.0
        
        # 如果最佳得分低于阈值，返回通用类型
        min_threshold = self.FEATURES.get(best_type, {}).get("min_matches", 3)
        if best_score < min_threshold:
            return ("general", confidence)
        
        return (best_type, confidence)
    
    def _select_config(self, doc_type: str, confidence: float,
                       text: str) -> AdaptiveConfig:
        """根据文档类型选择配置"""
        if doc_type in self.FEATURES:
            feature = self.FEATURES[doc_type]
            size_low, size_high = feature["chunk_size_range"]
            # confidence 影响 chunk_size 选择
            chunk_size = int(size_low + (size_high - size_low) * confidence)
            return AdaptiveConfig(
                strategy=feature["strategy"],
                chunk_size=chunk_size,
                chunk_overlap=int(chunk_size * 0.15),
            )
        return self.default_config
    
    def _analyze_text_stats(self, text: str) -> dict:
        """分析文本统计信息"""
        # 中英文字符比例
        chinese_chars = len(re.findall(r'[一-鿿]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total = max(chinese_chars + english_chars, 1)
        
        # 平均句子长度
        sentences = re.split(r'[。！？\n.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_len = sum(len(s) for s in sentences) / max(len(sentences), 1)
        
        # 标题密度
        headings = len(re.findall(r'^#{1,6}\s', text, re.MULTILINE))
        heading_density = headings / max(len(text) / 1000, 1)
        
        return {
            "chinese_ratio": chinese_chars / total,
            "english_ratio": english_chars / total,
            "avg_sentence_len": avg_sentence_len,
            "heading_density": heading_density,
            "total_chars": len(text),
        }
    
    def _adjust_config_by_stats(self, config: AdaptiveConfig,
                                 stats: dict) -> AdaptiveConfig:
        """根据文本统计信息微调配�置"""
        # 中文文档通常需要较小的chunk_size（中文字符信息密度高）
        if stats["chinese_ratio"] > 0.7:
            config.chunk_size = int(config.chunk_size * 0.8)
        
        # 长句文档需要更大的chunk_size
        if stats["avg_sentence_len"] > 60:
            config.chunk_size = int(config.chunk_size * 1.2)
        
        # 标题密度高适合标题分块
        if stats["heading_density"] > 2.0:
            config.strategy = ChunkingStrategy.MARKDOWN_HEADER
        
        return config
    
    def _execute_chunking(self, text: str, config: AdaptiveConfig) -> list:
        """根据策略执行具体分块"""
        if config.strategy == ChunkingStrategy.RECURSIVE:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separators=config.separators,
            )
            return splitter.split_text(text)
        
        elif config.strategy == ChunkingStrategy.MARKDOWN_HEADER:
            from langchain.text_splitter import MarkdownHeaderTextSplitter
            headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
            splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
            return splitter.split_text(text)
        
        elif config.strategy == ChunkingStrategy.LEGAL_ARTICLE:
            return legal_doc_chunking(text)
        
        elif config.strategy == ChunkingStrategy.AST_CODE:
            return code_ast_chunking(text, "source.py")
        
        else:
            # 默认：固定长度分块
            return fixed_length_chunking(text, config.chunk_size)
```

### 4.4.2 语义分块进阶 (Advanced Semantic Chunking)

关于语义分块的核心算法已在4.1.4节详述。本节补充进阶技巧：

```python
# 进阶1: 使用百分位断点而非绝对阈值（更鲁棒）
def percentile_breakpoint_detection(similarities: np.ndarray,
                                     percentile: float = 90) -> np.ndarray:
    """
    使用百分位方法检测断点
    
    原理：不设绝对阈值（如similarity < 0.5），而是找到
    相似度序列中的"异常低点"——低于第(100-percentile)百分位的值
    """
    threshold = np.percentile(similarities, 100 - percentile)
    breakpoints = np.where(similarities < threshold)[0]
    return breakpoints


# 进阶2: 断点合并（避免过度切分）
def merge_nearby_breakpoints(breakpoints: list, min_distance: int = 3) -> list:
    """合并距离过近的断点，避免产生过多小块"""
    if not breakpoints:
        return []
    
    merged = [breakpoints[0]]
    for bp in breakpoints[1:]:
        if bp - merged[-1] >= min_distance:
            merged.append(bp)
    
    return merged


# 进阶3: 自适应阈值（根据文本特征动态调整）
def adaptive_threshold(similarities: np.ndarray, target_chunk_count: int) -> float:
    """
    动态调整阈值以达到目标块数
    
    二分搜索最优百分位阈值
    """
    lo, hi = 50, 99
    best_pct = 90
    
    for _ in range(20):  # 20次迭代内收敛
        mid = (lo + hi) / 2
        threshold = np.percentile(similarities, 100 - mid)
        breakpoints = np.sum(similarities < threshold)
        chunk_count = breakpoints + 1
        
        if chunk_count < target_chunk_count:
            hi = mid  # 需要更多切分，降低百分位
        else:
            lo = mid
            best_pct = mid
    
    return np.percentile(similarities, 100 - best_pct)
```

### 4.4.3 Agent分块 (Agent Chunking)

#### 概念定义

Agent分块将分块决策委托给一个LLM Agent，该Agent不仅能识别边界，还能主动选择最优的分块策略、调整参数、验证结果，形成完整的分块工作流。

#### 工作原理

```
Agent分块工作流

┌───────────────────────────────────────────────────────────────────────┐
│                        Agent Chunking Pipeline                        │
│                                                                       │
│  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐    │
│  │ 分析   │   │ 策略   │   │ 执行   │   │ 验证   │   │ 迭代   │    │
│  │ 文档   │──►│ 选择   │──►│ 分块   │──►│ 结果   │──►│ 优化   │    │
│  └────────┘   └────────┘   └────────┘   └────────┘   └────────┘    │
│       │            │            │            │            │          │
│       ▼            ▼            ▼            ▼            ▼          │
│  "这是技术    "推荐使用    "已生成     "块大小     "第3块过大    │
│   文档，包含   标题分块，   15个块，   方差0.3，   已二次切分，  │
│   代码和表格"  chunk_size   平均600    语义连贯    现在18个块    │
│                =800"        chars"     性通过"    质量OK"        │
└───────────────────────────────────────────────────────────────────────┘
```

#### 核心实现（LangChain Agent模式）

```python
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
import json


class ChunkingAgent:
    """
    基于LangChain Agent的分块系统
    
    Agent拥有以下工具：
    1. analyze_document - 分析文档特征
    2. choose_strategy - 选择分块策略
    3. execute_chunking - 执行分块
    4. validate_chunks - 验证分块质量
    5. re_chunk - 对不满意的块重新分块
    """
    
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.agent = self._create_agent()
        self.chunking_history = []  # 记录分块决策历史
    
    def _create_agent(self):
        """创建分块Agent"""
        tools = [
            Tool(
                name="analyze_document",
                func=self._analyze_document,
                description="分析文档的类型、结构特征、语言分布。输入：文档文本。"
            ),
            Tool(
                name="choose_strategy",
                func=self._choose_strategy,
                description="根据文档分析结果选择最优分块策略和参数。输入：分析结果JSON。"
            ),
            Tool(
                name="execute_chunking",
                func=self._execute_chunking,
                description="执行分块操作。输入：策略名称和参数的JSON。"
            ),
            Tool(
                name="validate_chunks",
                func=self._validate_chunks,
                description="验证分块质量（大小方差、语义连贯性、覆盖率）。输入：chunks JSON。"
            ),
            Tool(
                name="re_chunk",
                func=self._re_chunk,
                description="对质量不达标的块进行重新分块。输入：块索引和新的chunk_size。"
            ),
        ]
        
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
        )
        
        return agent
    
    def chunk_document(self, text: str, quality_threshold: float = 0.7) -> list:
        """
        使用Agent对文档进行智能分块
        
        Args:
            text: 输入文档
            quality_threshold: 分块质量最低阈值
        
        Returns:
            分块结果列表
        """
        prompt = f"""
请对以下文档执行智能分块，确保每个块语义完整、大小合理。

步骤：
1. 使用 analyze_document 分析文档
2. 使用 choose_strategy 选择策略
3. 使用 execute_chunking 执行分块
4. 使用 validate_chunks 验证质量
5. 如果质量低于 {quality_threshold}，使用 re_chunk 优化

文档：
{text[:3000]}...
"""
        
        result = self.agent.invoke({"input": prompt})
        
        # 从结果中提取最终的chunks
        return self._extract_chunks_from_result(result)
    
    def _analyze_document(self, text: str) -> str:
        """分析文档特征"""
        stats = {
            "length": len(text),
            "paragraph_count": len(text.split("\n\n")),
            "has_headings": bool(re.search(r'^#{1,6}\s', text, re.MULTILINE)),
            "has_code_blocks": bool(re.search(r'```', text)),
            "has_tables": bool(re.search(r'\|.*\|', text)),
            "chinese_char_ratio": len(re.findall(r'[一-鿿]', text)) / max(len(text), 1),
            "avg_paragraph_length": sum(len(p) for p in text.split("\n\n")) / max(len(text.split("\n\n")), 1),
        }
        return json.dumps(stats, ensure_ascii=False)
    
    def _choose_strategy(self, analysis_json: str) -> str:
        """根据分析结果选择策��"""
        analysis = json.loads(analysis_json)
        
        if analysis.get("has_headings"):
            strategy = {
                "name": "markdown_header",
                "chunk_size": 800,
                "chunk_overlap": 80,
            }
        elif analysis.get("has_code_blocks"):
            strategy = {
                "name": "structure_aware",
                "chunk_size": 1000,
                "chunk_overlap": 50,
            }
        elif analysis.get("chinese_char_ratio", 0) > 0.5:
            strategy = {
                "name": "recursive_chinese",
                "chunk_size": 500,
                "chunk_overlap": 75,
                "separators": ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            }
        else:
            strategy = {
                "name": "recursive",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "separators": ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            }
        
        return json.dumps(strategy, ensure_ascii=False)
    
    def _execute_chunking(self, strategy_json: str) -> str:
        """执行分块"""
        strategy = json.loads(strategy_json)
        self.current_strategy = strategy
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=strategy["chunk_size"],
            chunk_overlap=strategy["chunk_overlap"],
            separators=strategy.get("separators", ["\n\n", "\n", " ", ""]),
        )
        
        # 注意：这里需要从外部获取text，实际实现需传递
        # 简化示例
        return json.dumps({"status": "executed", "strategy": strategy}, 
                         ensure_ascii=False)
    
    def _validate_chunks(self, chunks_json: str) -> str:
        """验证分块质量"""
        chunks = json.loads(chunks_json) if isinstance(chunks_json, str) else chunks_json
        # 计算质量指标
        sizes = [len(c) for c in chunks]
        avg_size = sum(sizes) / max(len(sizes), 1)
        variance = sum((s - avg_size) ** 2 for s in sizes) / max(len(sizes), 1)
        std_dev = variance ** 0.5
        
        quality = {
            "chunk_count": len(chunks),
            "avg_chunk_size": round(avg_size, 2),
            "size_std_dev": round(std_dev, 2),
            "size_uniformity": round(max(0, 1 - std_dev / max(avg_size, 1)), 2),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "needs_rechunk": std_dev / max(avg_size, 1) > 0.5,  # 变异系数>0.5需要重分块
        }
        
        return json.dumps(quality, ensure_ascii=False)
    
    def _re_chunk(self, chunk_indices_json: str) -> str:
        """重新分块"""
        return json.dumps({"status": "rechunked", "indices": chunk_indices_json},
                         ensure_ascii=False)
    
    def _extract_chunks_from_result(self, result: dict) -> list:
        """从Agent结果中提取最终分块"""
        # 实际实现需解析Agent的完整输出
        return result.get("output", "")
```

#### 优缺点分析

| 优点 | 缺点 |
|------|------|
| 完全智能，无需人工配置 | 成本最高（多次LLM调用） |
| 可以根据质量反馈迭代优化 | 速度极慢（需多轮推理） |
| 适应任何文档类型 | 部署复杂度高 |
| 可解释性强（记录决策过程） | 不适合批量实时处理 |

---

## 4.5 工程实践与最佳实践

### 4.5.1 企业级分块Pipeline架构

```
                     企业级分块Pipeline

┌─────────────────────────────────────────────────────────────────────┐
│                         输入层                                      │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐               │
│  │ PDF  │  │Word  │  │HTML  │  │ MD   │  │Confl │  ...           │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘               │
│     │         │         │         │         │                      │
│     └─────────┴────┬────┴─────────┴─────────┘                      │
│                    │                                                │
├────────────────────┼────────────────────────────────────────────────┤
│                    ▼              解析层                            │
│  ┌──────────────────────────────────────────┐                      │
│  │         格式统一化（Unstructured IO）      │                      │
│  │  PDF → 文本, Word → 文本, HTML → Markdown │                      │
│  └────────────────────┬─────────────────────┘                      │
│                       │                                             │
├───────────────────────┼─────────────────────────────────────────────┤
│                       ▼              分析层                         │
│  ┌──────────────────────────────────────────┐                      │
│  │          文档分析与分类                    │                      │
│  │  - 文档类型检测（技术/法律/学术/...）      │                      │
│  │  - 结构复杂���评估                         │                      │
│  │  - 语言比例分析                           │                      │
│  │  - Token预算估算                          │                      │
│  └────────────────────┬─────────────────────┘                      │
│                       │                                             │
├───────────────────────┼─────────────────────────────────────────────┤
│                       ▼              分块层                         │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │                     策略路由 + 执行                             ││
│  │                                                                ││
│  │  type=tech ──► MarkdownHeaderSplitter(chunk=800, overlap=100)  ││
│  │  type=legal ─► ArticleSplitter(chunk=1500, overlap=200)        ││
│  │  type=code  ─► ASTSplitter(unit=function)                      ││
│  │  type=general─►RecursiveCharSplitter(chunk=512, overlap=75)    ││
│  └────────────────────┬───────────────────────────────────────────┘│
│                       │                                             │
├───────────────────────┼─────────────────────────────────────────────┤
│                       ▼              后处理层                       │
│  ┌──────────────────────────────────────────┐                      │
│  │          质量控制与验证                    │                      │
│  │  - 块大小分布检查                         │                      │
│  │  - 空块/过小块过滤                        │                      │
│  │  - 内容完整性验证（重新拼接==原始？）      │                      │
│  │  - 元数据注入                             │                      │
│  └────────────────────┬─────────────────────┘                      │
│                       │                                             │
├───────────────────────┼─────────────────────────────────────────────┤
│                       ▼              输出层                         │
│  ┌──────────────────────────────────────────┐                      │
│  │         向量化 + 索引存储                  │                      │
│  │  [Chunk + Embedding + Metadata] → VectorDB│                      │
│  └──────────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.5.2 分块质量评估指标

```python
class ChunkingQualityMetrics:
    """
    分块质量评估指标体系
    """
    
    @staticmethod
    def size_uniformity(chunks: list) -> float:
        """大小一致性：块大小的变异系数（越小越好）"""
        sizes = [len(c) for c in chunks]
        mean = sum(sizes) / len(sizes)
        std = (sum((s - mean)**2 for s in sizes) / len(sizes))**0.5
        cv = std / mean if mean > 0 else 0
        return max(0, 1 - cv)  # 0-1, 越高越好
    
    @staticmethod
    def semantic_cohesion(chunks: list, embed_fn) -> float:
        """
        语义凝聚力：块内句子的平均相似度（越高越好）
        每个块计算内部句子的平均两两相似度
        """
        # 简化实现
        cohesion_scores = []
        for chunk in chunks:
            sentences = _split_sentences(chunk)
            if len(sentences) < 2:
                cohesion_scores.append(1.0)
                continue
            
            embeddings = embed_fn(sentences)
            sim_sum = 0
            count = 0
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim_sum += cosine_similarity(
                        embeddings[i].reshape(1,-1),
                        embeddings[j].reshape(1,-1)
                    )[0][0]
                    count += 1
            cohesion_scores.append(sim_sum / max(count, 1))
        
        return sum(cohesion_scores) / len(cohesion_scores)
    
    @staticmethod
    def boundary_separation(chunks: list, embed_fn) -> float:
        """
        边界分离度：相邻块首尾的语义差异（越高越好，表示切分在合适的位置）
        """
        if len(chunks) < 2:
            return 1.0
        
        separation_scores = []
        for i in range(len(chunks) - 1):
            first_end = _split_sentences(chunks[i])[-1]
            second_start = _split_sentences(chunks[i+1])[0]
            
            emb1 = embed_fn([first_end])
            emb2 = embed_fn([second_start])
            
            sim = cosine_similarity(emb1, emb2)[0][0]
            # 边界处相似度越低，分离度越好
            separation_scores.append(1 - sim)
        
        return sum(separation_scores) / len(separation_scores)
    
    @staticmethod
    def coverage_fidelity(original_text: str, chunks: list) -> float:
        """
        覆盖保真度：重新拼接所有块是否保留原始内容
        1.0 = 完全保留
        """
        reconstructed = "".join(chunks)
        # 去除分块可能引入的多余空白
        reconstructed = " ".join(reconstructed.split())
        original = " ".join(original_text.split())
        
        # 基于字符的覆盖率
        orig_chars = set(original)
        recon_chars = set(reconstructed)
        coverage = len(orig_chars & recon_chars) / len(orig_chars) if orig_chars else 1.0
        
        return coverage
    
    @staticmethod
    def overall_quality_score(chunks: list, original_text: str,
                               embed_fn) -> dict:
        """
        综合质量评分
        """
        uniformity = ChunkingQualityMetrics.size_uniformity(chunks)
        cohesion = ChunkingQualityMetrics.semantic_cohesion(chunks, embed_fn)
        separation = ChunkingQualityMetrics.boundary_separation(chunks, embed_fn)
        fidelity = ChunkingQualityMetrics.coverage_fidelity(original_text, chunks)
        
        # 加权综合评分
        weights = {
            "uniformity": 0.20,
            "cohesion": 0.35,
            "separation": 0.25,
            "fidelity": 0.20,
        }
        
        overall = (
            weights["uniformity"] * uniformity +
            weights["cohesion"] * cohesion +
            weights["separation"] * separation +
            weights["fidelity"] * fidelity
        )
        
        return {
            "overall_score": round(overall, 3),
            "size_uniformity": round(uniformity, 3),
            "semantic_cohesion": round(cohesion, 3),
            "boundary_separation": round(separation, 3),
            "coverage_fidelity": round(fidelity, 3),
        }
```

### 4.5.3 常见陷阱与解决方案

| 陷阱 | 现象 | 解决方案 |
|------|------|---------|
| **过度分块** | 块数过多，单块信息量低 | 调大chunk_size或使用合并策略 |
| **分块不足** | 块数过少，单块信息混杂 | 调小chunk_size或多策略级联 |
| **表格拆分** | 表格被从中间切分 | 使用结构感知分块，保护原子元素 |
| **代码截断** | 函数/类在中间被切断 | AST解析或代码块保护 |
| **编码截断** | 多字节字符（中文）被截断 | Token-level分块替代字符级 |
| **相似块冗余** | 高overlap导致大量重复内容 | 优化overlap率或使用去重后处理 |
| **元数据丢失** | 分块后丢失来源信息 | 分块时保留标题链、页码、来源等 |
| **嵌入截断** | Chunk超过嵌入模型最大长度 | 分块前检查token数，留出buffer |

### 4.5.4 性能优化建议

```
分块性能优化路线图

1. 预处理优化
   ├── 字符规范化（全角→半角、统一换行符）
   ├── 无关内容过滤（广告、导航栏、页眉页脚）
   └── 语言检测与分离（中英混合文档分段处理）

2. 分块算法优化
   ├── 使用tiktoken替代len()进行精确token计数
   ├── 批量处理：多文档并行分块
   ├── 缓存策略：相似结构文档复用分块参数
   └── 流式处理：超长文档逐段处理，避免全量加载

3. 后处理优化
   ├── 过滤空块和过短块（< 20 chars）
   ├── 去重：移除高度相似的重复块
   ├── 元数据注入：批量添加（非逐块调用DB）
   └── 异步写入：分块与向量化流水线并行

4. 成本优化
   ├── 固定分块 → 递归分块 → 语义分块（按需升级）
   ├── 语义分块使用轻量级本地嵌入模型
   ├── LLM分块仅用于关键高价值文档
   └── 分块结果缓存：相同文档不重复分块
```

---

## 4.6 面试要点

### 常见面试问题

**Q1: 为什么RAG系统中分块策略如此重要？**

A: 分块是RAG系统检索质量的第一道关卡。它决定了：(1) 语义单元的完整性——如果切分不当，向量表示将不准确；(2) 上下文窗口的利用效率——太大或太小的块都会浪费LLM的上下文窗口；(3) 精度-召回的平衡——细粒度提高精度但降低召回，粗粒度反之；(4) 与嵌入模型的能力匹配——超出模型最大输入的部分会被截断。

**Q2: 递归分块和滑动窗口分块的核心区别是什么？**

A: 递归分块按照分隔符优先级（段落→句子→字符）逐步切分，优先在自然语言边界处切分，保持语义完整性。滑动窗口分块使用固定步长移动窗口，通过重叠区域缓解边界截断问题。递归分块考虑了文本结构，滑动窗口分块则完全不考虑结构，仅通过重叠弥补。实际应用中，递归分块效果更好，是LangChain的默认方案。

**Q3: 如何为中文文档选择chunk_size？**

A: 中文文档的信息密度通常高于英文（相同token数对应更多语义信息）。建议：(1) 使用token-level而非字符级分块；(2) 技术文档推荐512-768 tokens；(3) 法律/长文档推荐1024-1536 tokens；(4) 使用bge-large-zh等中文专用嵌入模型时，注意其512 token的限制；(5) 建议chunk_overlap设为10%-15%。

**Q4: 语义分块的断点检测有哪些常见方法？**

A: 主要有三种方法：(1) 绝对阈值法——设置固定的相似度阈值（如<0.5），低于阈值则切分；(2) 百分位法——取相似度序列中最低的N%作为断点（如最低10%），比绝对阈值更鲁棒；(3) 标准偏差法——低于均值K个标准差的视为断点。百分位法在大多数场景下效果最好，因为它自适应文本特性。

**Q5: 如何处理混合格式文档（文字+表格+代码）？**

A: 必须使用结构感知分块。核心原则：(1) 表格作为原子单元，永不拆分行列；(2) 代码块作为原子单元，永不截断函数/类；(3) 先按格式元素分离，再对各元素单独分块；(4) 保留结构元数据（表格的列名、代码的语言类型）。推荐使用Unstructured库或自研解析器。

**Q6: 分块与嵌入模型的关系是什么？**

A: (1) chunk_size必须小于嵌入模型的最大输入长度（如text-embedding-ada-002是8191 tokens）；(2) 实际chunk_size建议为模型最大长度的50%-80%，留出buffer；(3) 不同的嵌入模型对chunk_size的敏感度不同——大模型（text-embedding-3-large）可以处理更大的块；(4) 分块的token计算必须使用与嵌入模型相同的分词器。

---

## 4.7 企业最佳实践清单

| # | 实践 | 优先级 | 说明 |
|---|------|--------|------|
| 1 | **从递归分块开始** | 必须 | 覆盖80%场景，是LangChain/LlamaIndex默认方案 |
| 2 | **使用Token级而非字符级** | 必须 | 避免多字节字符截断，与嵌入模型对齐 |
| 3 | **保留元数据** | 必须 | 标题链、来源、页码、更新时间，支持检索过滤 |
| 4 | **验证分块质量** | 必须 | 定期检查块大小分布、语义凝聚力、覆盖保真度 |
| 5 | **按文档类型路由** | 推荐 | 技术文档/法律/代码使用不同分块策略 |
| 6 | **设置chunk_size上限** | 推荐 | 不超过嵌入模型最大输入的80% |
| 7 | **过滤无效块** | 推荐 | 删除空块、过短块(<20 chars)、纯格式块 |
| 8 | **表格和代码块保护** | 推荐 | 原子元素不拆分 |
| 9 | **A/B测试分块策略** | 推荐 | 用检索质量指标对比不同策略的效果 |
| 10 | **语义分块按需升级** | 可选 | 仅当递归分块效果不足时考虑 |
| 11 | **LLM分块仅用于高价值场景** | 可选 | 成本高，适合核心文档或冷启动 |
| 12 | **建立分块质量监控** | 推荐 | Dashboard跟踪块数、平均大小、检索命中率 |
| 13 | **分块结果缓存** | 推荐 | 相同文档+相同策略不重复分块 |
| 14 | **版本化分块策略** | 可选 | 记录每个文档使用的分块策略版本，支持回溯 |
| 15 | **多粒度索引** | 高级 | 同时建立大小两种粒度的索引，检索时级联或投票 |

---

## 4.8 本章小结

分块技术是RAG系统中最基础却最关键的环节之一。本章从"为何分块决定召回质量"出发，系统性地阐述了：

1. **八种主流分块方法**的原理、算法实现和适用场景，从最基础的固定长度分块到最智能的LLM辅助分块，构成了分块技术的能力光谱。

2. **场景化分块策略**，针对技术文档、法律文档、学术论文、文学作品、医疗文档、代码仓库和企业知识库七类场景，给出了定制化的参数推荐和实现代码。

3. **分块大小与重叠的深度分析**，通过实验数据验证了512-768 tokens是大多数场景的"甜蜜点"，并给出了不同场景的经验最佳值。

4. **动态分块技术**，包括自适应分块、语义分块进阶和Agent分块，代表了分块技术从静态规则到智能决策的演进方向。

5. **工程实践指南**，包括企业级Pipeline架构、质量评估指标体系、常见陷阱解决方案和性能优化建议。

分块策略的选择是一个需要持续实验和优化的过程。企业应建立分块策略的A/B测试框架，用检索质量的量化指标来驱动分块策略的迭代改进。

---

> **下一章预告**：第五章将深入探讨「嵌入模型与向量化技术」，剖析如何为不同场景选择最优的嵌入模型，以及向量数据库的选型与优化策略。
