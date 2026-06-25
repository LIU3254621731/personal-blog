# 第九章：上下文构建与提示工程 (Context Construction & Prompt Engineering)

## 9.1 章节概述

检索增强生成（RAG）系统的核心链路可概括为"检索-构建-生成"三步。在上一章深入探讨检索策略之后，本章聚焦于从检索结果到大模型最终输出的关键环节——上下文构建（Context Construction）与提示工程（Prompt Engineering）。这两个环节决定了检索到的知识能否被大模型有效利用，直接影响生成结果的准确性、忠实度和完整度。

本章将系统阐述以下内容：
- 如何将检索到的文档片段高效地打包进LLM上下文窗口
- 如何设计提示模板以引导模型正确利用检索到的知识
- 如何应对上下文长度限制、提示污染等关键技术挑战
- 企业级实践中的提示版本管理、质量评估与动态选择策略

---

## 9.2 上下文构建 (Context Construction)

### 9.2.1 概念定义与背景

**定义**：上下文构建是指将检索系统返回的文档片段（Chunks）按照特定策略进行组织、压缩和格式化，转化为大语言模型可理解的结构化上下文输入的过程。

**背景**：早期RAG系统简单地将检索结果拼接后传入模型，忽略了上下文的结构化组织。随着应用深入，人们发现以下几点关键问题：
1. 检索结果中存在冗余、矛盾和无用信息，简单拼接会"污染"上下文
2. 大模型的上下文窗口有限（从早期的4K到如今的128K+），需要精细化的Token预算管理
3. 不同位置的上下文信息对模型输出的影响权重不同（首尾偏好效应）

**解决的问题**：
- Token利用率低下导致有效信息密度不足
- 上下文顺序不当导致模型注意力分散
- 上下文过长导致推理成本线性增长
- 多轮对话中上下文膨胀和遗忘

---

### 9.2.2 上下文打包策略 (Context Packing)

上下文打包是将检索到的文档片段高效编排进LLM上下文的核心技术。好的打包策略能显著提升信息密度和答案质量。

#### 9.2.2.1 基本打包格式

标准RAG上下文打包通常采用以下结构：

```
[系统指令 (System Prompt)]
  |-- 角色定义
  |-- 行为约束
  |-- 输出格式要求

[检索上下文 (Retrieved Context)]
  |-- 文档1: [标题] [来源] [相关性得分]
  |-- 文档2: [标题] [来源] [相关性得分]
  |-- ...

[对话历史 (Conversation History)]  (多轮场景)
  |-- 用户: ...
  |-- 助手: ...

[当前查询 (Current Query)]
  |-- 用户当前问题和要求
```

#### 9.2.2.2 上下文打包实现

下面是上下文打包的Python参考实现：

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json

@dataclass
class ContextDocument:
    """上下文文档片段"""
    content: str
    doc_id: str
    title: Optional[str] = None
    source: Optional[str] = None
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: Optional[int] = None


class ContextPacker:
    """
    上下文打包器：将检索到的文档片段组织为LLM可用的上下文格式
    
    核心职责：
    1. 去重与冗余处理
    2. Token预算分配
    3. 格式化输出
    """
    
    def __init__(self, token_counter, max_context_tokens: int = 4000):
        self.token_counter = token_counter
        self.max_context_tokens = max_context_tokens
    
    def estimate_tokens(self, text: str) -> int:
        """估算文本的Token数量"""
        return self.token_counter(text)
    
    def deduplicate(self, documents: List[ContextDocument], 
                    threshold: float = 0.85) -> List[ContextDocument]:
        """
        基于内容的文档去重
        
        采用Jaccard相似度进行快速去重，避免相同/高度相似的内容
        重复占用宝贵的上下文空间。
        """
        if not documents:
            return []
        
        def jaccard_similarity(a: str, b: str) -> float:
            set_a = set(a.lower().split())
            set_b = set(b.lower().split())
            if not set_a or not set_b:
                return 0.0
            return len(set_a & set_b) / len(set_a | set_b)
        
        kept = []
        for doc in documents:
            is_dup = False
            for existing in kept:
                sim = jaccard_similarity(doc.content, existing.content)
                if sim >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(doc)
        
        return kept
    
    def pack(self, documents: List[ContextDocument], 
             query: str,
             include_scores: bool = True,
             format_type: str = "detailed") -> str:
        """
        将文档列表打包为上下文字符串
        
        format_type:
        - "detailed": 包含标题、来源、相关性得分的完整格式
        - "minimal": 仅包含内容的最小格式
        - "xml": XML标签格式（推荐用于指令遵循能力强的模型）
        """
        # 去重
        documents = self.deduplicate(documents)
        
        # 按相关性降序排列
        documents.sort(key=lambda x: x.relevance_score, reverse=True)
        
        packed_chunks = []
        current_tokens = 0
        
        for i, doc in enumerate(documents):
            if format_type == "detailed":
                chunk = self._format_detailed(doc, i + 1, include_scores)
            elif format_type == "minimal":
                chunk = self._format_minimal(doc)
            elif format_type == "xml":
                chunk = self._format_xml(doc, i + 1, include_scores)
            else:
                chunk = self._format_detailed(doc, i + 1, include_scores)
            
            chunk_tokens = self.estimate_tokens(chunk)
            if current_tokens + chunk_tokens > self.max_context_tokens:
                break
            
            packed_chunks.append(chunk)
            current_tokens += chunk_tokens
        
        # 组装最终上下文
        header = self._build_header(len(packed_chunks), len(documents), query)
        return header + "\n\n".join(packed_chunks)
    
    def _format_detailed(self, doc: ContextDocument, index: int, 
                         include_scores: bool) -> str:
        """详细格式：包含完整元数据"""
        parts = [f"[文档 {index}]"]
        if doc.title:
            parts.append(f"标题: {doc.title}")
        if doc.source:
            parts.append(f"来源: {doc.source}")
        if include_scores and doc.relevance_score > 0:
            parts.append(f"相关度: {doc.relevance_score:.3f}")
        parts.append(f"\n{doc.content.strip()}")
        return "\n".join(parts)
    
    def _format_minimal(self, doc: ContextDocument) -> str:
        """最小格式：仅内容"""
        return doc.content.strip()
    
    def _format_xml(self, doc: ContextDocument, index: int, 
                    include_scores: bool) -> str:
        """XML标签格式：结构化上下文，便于模型解析"""
        attrs = f'id="{doc.doc_id}" index="{index}"'
        if include_scores and doc.relevance_score > 0:
            attrs += f' relevance="{doc.relevance_score:.3f}"'
        
        title_tag = f"<title>{doc.title}</title>\n" if doc.title else ""
        source_tag = f"<source>{doc.source}</source>\n" if doc.source else ""
        
        return (
            f'<document {attrs}>\n'
            f'{title_tag}'
            f'{source_tag}'
            f'<content>\n{doc.content.strip()}\n</content>\n'
            f'</document>'
        )
    
    def _build_header(self, packed: int, total: int, query: str) -> str:
        """构建上下文头部说明"""
        return (
            f"以下是与用户问题相关的 {packed} 个参考文档片段"
            f"（从总共 {total} 个检索结果中筛选）：\n\n"
        )
```

---

### 9.2.3 上下文排序策略 (Context Ordering Strategies)

检索到的文档片段如何排序，直接影响LLM的注意力分配和最终生成质量。以下是三种主流排序策略：

#### 9.2.3.1 相关性降序排列 (Relevance-Based Descending)

```
+-------------------------------------------------------+
|  排列方式：score=0.97 → 0.85 → 0.72 → 0.61 → 0.43   |
+-------------------------------------------------------+
|  优点：最相关的内容在窗口开头，充分利用模型的       |
|        "首部偏好" (Primacy Bias)                       |
|  缺点：尾部内容可能被"遗忘"；可能造成信息重复       |
|        （多个高分片段可能来自同一文档的相邻部分）    |
|  适用：事实型查询、短上下文场景                       |
+-------------------------------------------------------+
```

#### 9.2.3.2 时间顺序排列 (Chronological)

```
+-------------------------------------------------------+
|  排列方式：2020-Q1 → 2021-Q3 → 2022-Q2 → 2023-Q4     |
+-------------------------------------------------------+
|  优点：保留事件的时间演化关系，适合时间相关的查询   |
|  缺点：最近的信息可能在窗口末尾被忽略                |
|  适用：历史查询、事件追踪、版本演进、时间线分析      |
+-------------------------------------------------------+
```

#### 9.2.3.3 分层排序 (Hierarchical)

```
+-------------------------------------------------------+
|  第一层（窗口开头20%）：高相关度核心文档              |
|  第二层（窗口中部60%）：中等相关度补充文档            |
|  第三层（窗口末尾20%）：背景/上下文补充文档           |
+-------------------------------------------------------+
|  优点：兼顾首部偏好和尾部唤醒效应（Recency Bias），  |
|        核心信息优先、补充信息不缺失                   |
|  缺点：排序逻辑复杂，需要额外的分层策略              |
|  适用：综合分析型查询、长上下文场景                   |
+-------------------------------------------------------+
```

**排序策略选择指南**：

| 查询类型 | 推荐策略 | 原因 |
|---------|---------|------|
| 事实型查询（"X的定义是什么"） | 相关性降序 | 需要精确匹配的信息片段 |
| 时间相关查询（"近年来的变化"） | 时间顺序 | 保留时间演化关系 |
| 综合分析（"对比分析X和Y"） | 分层排序 | 兼顾多维度信息 |
| 法律/合规查询 | 分层+时间混合 | 需覆盖法规历史与现状 |
| 技术故障排查 | 相关性降序 | 快速定位关键错误信息 |

#### 9.2.3.4 排序策略实现

```python
from enum import Enum
from typing import Callable

class OrderStrategy(Enum):
    RELEVANCE_DESC = "relevance_desc"
    RELEVANCE_ASC = "relevance_asc"  
    CHRONOLOGICAL = "chronological"
    HIERARCHICAL = "hierarchical"
    LOST_IN_MIDDLE = "lost_in_middle"  # 解决"Lost in Middle"问题


class ContextOrderer:
    """
    上下文排序器
    
    解决的核心问题——"Lost in the Middle"效应：
    研究表明，LLM在处理长上下文时，对开头和结尾的内容关注度最高，
    对中间部分的内容关注度显著下降。通过合理的排序策略，
    可以确保重要信息不在模型的"注意力低谷"区域。
    """
    
    def __init__(self, strategy: OrderStrategy = OrderStrategy.RELEVANCE_DESC):
        self.strategy = strategy
    
    def order(self, documents: List[ContextDocument], 
              query: str = "") -> List[ContextDocument]:
        """对文档列表进行排序"""
        if self.strategy == OrderStrategy.RELEVANCE_DESC:
            return self._order_relevance_desc(documents)
        elif self.strategy == OrderStrategy.RELEVANCE_ASC:
            return self._order_relevance_asc(documents)
        elif self.strategy == OrderStrategy.CHRONOLOGICAL:
            return self._order_chronological(documents)
        elif self.strategy == OrderStrategy.HIERARCHICAL:
            return self._order_hierarchical(documents)
        elif self.strategy == OrderStrategy.LOST_IN_MIDDLE:
            return self._order_lost_in_middle(documents)
        return documents
    
    def _order_relevance_desc(self, docs):
        """相关性降序：高分在前如[0.97, 0.85, 0.72,...]"""
        return sorted(docs, key=lambda x: x.relevance_score, reverse=True)
    
    def _order_relevance_asc(self, docs):
        """相关性升序：将最相关的放在最后，利用尾部唤醒效应"""
        return sorted(docs, key=lambda x: x.relevance_score)
    
    def _order_chronological(self, docs):
        """按时间戳排序"""
        return sorted(docs, 
                     key=lambda x: x.metadata.get("timestamp", ""))
    
    def _order_hierarchical(self, docs):
        """分层排序：核心→补充→背景"""
        sorted_docs = sorted(docs, key=lambda x: x.relevance_score, reverse=True)
        n = len(sorted_docs)
        if n <= 3:
            return sorted_docs
        
        high = sorted_docs[:max(1, n//5)]       # 前20%
        mid = sorted_docs[max(1, n//5):-max(1, n//5)]  # 中间60%
        low = sorted_docs[-max(1, n//5):]       # 后20%
        
        # 高→中→低的结构
        return high + mid + low
    
    def _order_lost_in_middle(self, docs):
        """
        对抗"Lost in the Middle"效应的排序策略:
        
        将最重要的文档交替放在开头和结尾，
        次要文档放在中间，确保重要信息不在注意力低谷。
        
        排序结果示意：
        [核心1, 核心3, 补充_1, 补充_2, ..., 核心2, 核心4]
          ↑头部                             ↑尾部
        """
        sorted_docs = sorted(docs, key=lambda x: x.relevance_score, reverse=True)
        n = len(sorted_docs)
        if n <= 4:
            return sorted_docs
        
        # 将文档按重要程度分为核心、次要两组
        split = max(2, n // 3)
        core = sorted_docs[:split]
        secondary = sorted_docs[split:]
        
        # 核心文档交替分布在头尾
        result = []
        left, right = 0, len(core) - 1
        while left <= right:
            if left <= right:
                result.append(core[left])
                left += 1
            if left <= right:
                result.append(core[right])
                right -= 1
        
        # 次要文档填充中间
        mid_point = len(result) // 2
        return result[:mid_point] + secondary + result[mid_point:]
```

---

### 9.2.4 Token预算分配 (Token Budget Allocation)

在有限的上下文窗口中，合理的Token预算分配是保证RAG系统质量的关键。

#### 9.2.4.1 标准分配方案

```
+====================================================================+
|                    LLM 上下文窗口 (Context Window)                    |
|                                                                      |
|  +--------------------+--------------------------------------------+ |
|  |  系统提示           |  检索上下文 + 对话历史 + 用户查询 + 预留输出  | |
|  | (System Prompt)     |  (Dynamic Context)                          | |
|  +--------------------+--------------------------------------------+ |
|  |  ~500-1500 Tokens   |  ~2500-3500 Tokens (4K窗口)                 | |
|  |                     |  ~30K-80K Tokens (128K窗口)                 | |
|  |  ~10-15%            |  ~85-90%                                    | |
|  +--------------------+--------------------------------------------+ |
|======================================================================|
|  细分 (Dynamic Context 内部)：                                        |
|  +---------------------+---------------------+---------------------+ |
|  | 检索上下文          | 对话历史             | 用户查询 + 预留输出    | |
|  | (Retrieved Context) | (Conversation Hist)  | (Query + Reserve)   | |
|  +---------------------+---------------------+---------------------+ |
|  | 50-60%              | 20-25%               | 15-20%               | |
|  +---------------------+---------------------+---------------------+ |
+======================================================================+
```

#### 9.2.4.2 不同模型的Token分配参考

| 模型 | 上下文窗口 | 系统提示 | 检索上下文 | 对话历史 | 用户查询+输出 |
|------|-----------|---------|-----------|---------|-------------|
| GPT-3.5-Turbo | 4K | 500 | 2000-2500 | 500 | 500-1000 |
| GPT-3.5-Turbo-16K | 16K | 1000 | 10000-12000 | 2000 | 1000-2000 |
| GPT-4-Turbo | 128K | 2000 | 80000-100000 | 10000 | 8000-16000 |
| Claude-3.5-Sonnet | 200K | 2000 | 140000-160000 | 20000 | 8000-16000 |
| Gemini-1.5-Pro | 1M | 4000 | 700000-800000 | 50000 | 50000-100000 |
| DeepSeek-V3 | 128K | 2000 | 80000-100000 | 10000 | 8000-16000 |
| 开源模型 (Qwen2-7B) | 32K | 1000 | 20000-24000 | 3000 | 3000-5000 |

#### 9.2.4.3 Token预算管理器实现

```python
@dataclass
class TokenBudget:
    """Token预算配置"""
    total_limit: int
    system_prompt_ratio: float = 0.10
    context_ratio: float = 0.55
    history_ratio: float = 0.20
    query_output_ratio: float = 0.15
    
    @property
    def system_prompt_limit(self) -> int:
        return int(self.total_limit * self.system_prompt_ratio)
    
    @property
    def context_limit(self) -> int:
        return int(self.total_limit * self.context_ratio)
    
    @property
    def history_limit(self) -> int:
        return int(self.total_limit * self.history_ratio)
    
    @property
    def query_output_limit(self) -> int:
        return int(self.total_limit * self.query_output_ratio)


class TokenBudgetManager:
    """
    Token预算管理器
    
    职责：
    1. 根据上下文窗口大小动态分配各部分的Token配额
    2. 实时追踪已使用的Token数
    3. 在超出限制时触发压缩或截断策略
    """
    
    def __init__(self, budget: TokenBudget, token_counter: Callable):
        self.budget = budget
        self.token_counter = token_counter
        self.used = {
            "system": 0,
            "context": 0,
            "history": 0,
            "query": 0
        }
    
    def can_add_context(self, text: str) -> bool:
        """检查是否可以在不超预算的情况下添加上下文"""
        tokens = self.token_counter(text)
        return self.used["context"] + tokens <= self.budget.context_limit
    
    def add_context(self, text: str) -> bool:
        """尝试添加上下文文档，返回是否成功"""
        tokens = self.token_counter(text)
        if self.used["context"] + tokens > self.budget.context_limit:
            return False
        self.used["context"] += tokens
        return True
    
    def remaining_context_tokens(self) -> int:
        """剩余可用上下文Token数"""
        return max(0, self.budget.context_limit - self.used["context"])
    
    def get_usage_report(self) -> Dict[str, Any]:
        """获取Token使用报告"""
        total_used = sum(self.used.values())
        return {
            "total_limit": self.budget.total_limit,
            "total_used": total_used,
            "total_remaining": self.budget.total_limit - total_used,
            "utilization": f"{total_used / self.budget.total_limit * 100:.1f}%",
            "breakdown": self.used.copy()
        }
    
    def suggest_compression(self) -> bool:
        """判断是否需要触发上下文压缩"""
        utilization = sum(self.used.values()) / self.budget.total_limit
        return utilization > 0.85
```

---

### 9.2.5 上下文压缩 (Context Compression)

当检索结果过多或文档片段过长时，需要进行上下文压缩。上下文压缩在信息不丢失和不冗余之间寻求平衡。

#### 9.2.5.1 压缩策略对比

```
+=========================================================================+
|                        上下文压缩策略全景图                              |
+=========================================================================+
|                                                                          |
|  [1] LLMLingua系列 --- 基于小模型的信息密度压缩                         |
|      ┌─────────────────────────────────────────────────────────┐        |
|      │ 原始文本 → 困惑度计算 → 低信息密度词移除 → 压缩文本     │        |
|      │ LLMLingua: 2-5x压缩比, LLMLingua-2: 5-10x压缩比        │        |
|      │ 特点: Token级别的动态压缩，保留关键词和关键实体         │        |
|      └─────────────────────────────────────────────────────────┘        |
|                                                                          |
|  [2] 摘要式压缩 --- 使用LLM对文档进行摘要                               |
|      ┌─────────────────────────────────────────────────────────┐        |
|      │ 多文档 → Extractive Summary(提取式) → 精简上下文        │        |
|      │        → Abstractive Summary(生成式) → 高信息密度摘要   │        |
|      │ 特点: 信息密度最高, 但有信息丢失/失真的风险             │        |
|      └─────────────────────────────────────────────────────────┘        |
|                                                                          |
|  [3] 选择性压缩 --- 根据查询动态选择相关句子                            |
|      ┌─────────────────────────────────────────────────────────┐        |
|      │ 查询 + 文档 → Sentence Embedding → 句子级相关度评分      │        |
|      │ → Top-K相关句子 → 重组为精简文档                         │        |
|      │ 特点: 查询感知、保留相关细节、过滤无关信息              │        |
|      └─────────────────────────────────────────────────────────┘        |
|                                                                          |
|  [4] 递归压缩 --- 层次化压缩长文档                                      |
|      ┌─────────────────────────────────────────────────────────┐        |
|      │ 长文档 → 分段 → 每段摘要 → 摘要合并 → 最终上下文        │        |
|      │ 特点: 适合超长文档(>32K tokens)的处理                   │        |
|      └─────────────────────────────────────────────────────────┘        |
|                                                                          |
+=========================================================================+
```

#### 9.2.5.2 LLMLingua风格压缩实现

```python
class ContextCompressor:
    """
    上下文压缩器
    
    提供多种压缩策略：
    1. LLMLingua风格：基于困惑度的token级压缩
    2. 摘要式：基于LLM的摘要压缩
    3. 选择性：基于查询相关度的句子级压缩
    """
    
    def __init__(self, 
                 compression_model=None,
                 target_ratio: float = 0.5):
        self.model = compression_model
        self.target_ratio = target_ratio
    
    def compress_lingua_style(self, text: str, 
                               target_tokens: int) -> str:
        """
        LLMLingua风格的困惑度压缩
        
        原理：
        1. 使用小型语言模型（如GPT-2 small）计算每token困惑度
        2. 困惑度越低的token，信息密度通常越高
        3. 保留低困惑度token，移除高困惑度token
        4. 实现2-5倍的无损/微损压缩
        """
        # 简化的实现示意（实际需要加载小型LLM）
        words = text.split()
        
        # 模拟：基于词性保留关键词
        # 实际实现中使用小模型的困惑度评分
        keep_words = []
        for i, word in enumerate(words):
            # 保留：专有名词、数字、关键动词、否定词
            if (word[0].isupper() or word.isdigit() or 
                word.lower() in {'not', 'no', 'never', 'without'} or
                len(word) > 3):  # 过滤短功能词
                keep_words.append(word)
        
        compressed = ' '.join(keep_words)
        
        # 如果仍超长，进行截断
        if self._estimate_tokens(compressed) > target_tokens:
            compressed = self._truncate_by_tokens(compressed, target_tokens)
        
        return compressed
    
    def compress_selective(self, text: str, query: str, 
                           top_k_sentences: int = 5) -> str:
        """
        基于查询的选择性压缩
        
        流程：
        query + 文档 → 句子嵌入 → cosine相似度 → Top-K → 重组成文
        """
        import re
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 分句
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        if len(sentences) <= top_k_sentences:
            return text
        
        # 实际场景中使用embedding model
        # 这里展示核心逻辑
        query_words = set(query.lower().split())
        
        scored = []
        for sent in sentences:
            sent_words = set(sent.lower().split())
            if not sent_words:
                continue
            # 词汇重叠度作为相关性的近似
            overlap = len(query_words & sent_words) / len(sent_words)
            # 附加上下文位置权重（首段和尾段加权）
            scored.append((sent, overlap))
        
        # 按相关度排序选取Top-K
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [s for s, _ in scored[:top_k_sentences]]
        
        # 维持原始顺序
        original_order = {s: i for i, s in enumerate(sentences)}
        selected.sort(key=lambda s: original_order.get(s, 0))
        
        return ' '.join(selected)
    
    def compress_recursive(self, text: str, 
                           chunk_size: int = 1000,
                           summary_func: Callable = None) -> str:
        """
        递归压缩：适合超长文档
        
        流程：
        1. 分段：将长文本切分为chunk_size大小的段落
        2. 摘要：对每段生成摘要
        3. 合并：摘要合并，如果仍超长则递归
        """
        if self._estimate_tokens(text) <= chunk_size:
            return text
        
        # 分段
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_len = 0
        
        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len > chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_len = 0
            current_chunk.append(para)
            current_len += para_len
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        # 每段提取关键句（模拟摘要）
        summaries = []
        for chunk in chunks:
            sentences = chunk.split('. ')
            # 保留每段最重要的1-3个句子
            key_sentences = sentences[:min(3, len(sentences))]
            summaries.append('. '.join(key_sentences))
        
        combined = '\n\n'.join(summaries)
        
        # 递归：如果合并后仍超长
        return self.compress_recursive(combined, chunk_size, summary_func)
    
    def _estimate_tokens(self, text: str) -> int:
        """简易Token估算：英文按单词数*1.3，中文按字符数/1.5"""
        import re
        en_words = len(re.findall(r'[a-zA-Z]+', text))
        cn_chars = len(re.findall(r'[一-鿿]', text))
        return int(en_words * 1.3 + cn_chars / 1.5)
    
    def _truncate_by_tokens(self, text: str, max_tokens: int) -> str:
        """按Token数截断文本"""
        current = 0
        words = text.split()
        result = []
        for word in words:
            t = self._estimate_tokens(word)
            if current + t > max_tokens:
                break
            result.append(word)
            current += t
        return ' '.join(result)
```

#### 9.2.5.4 上下文压缩技术对比

| 压缩方法 | 压缩比 | 信息保真度 | 计算开销 | 最佳场景 |
|---------|-------|-----------|---------|---------|
| LLMLingua | 2-5x | 高 (90%+) | 中 (需小模型推理) | 通用场景，需保真 |
| LLMLingua-2 | 5-10x | 中高 (85%+) | 中 | 大量文档，可接受微损 |
| 摘要式 | 3-20x | 中 (70-85%) | 高 (需LLM调用) | 探索性查询，长文档 |
| 选择性 | 2-10x | 中高 (查询相关) | 低 (向量比较) | 查询明确的场景 |
| 递归压缩 | 4-50x | 中低 (多层次丢失) | 高 | 超长文档(>32K) |

---

### 9.2.6 多轮对话上下文管理

多轮对话场景中，上下文管理面临更大的挑战：需要在有限的窗口内同时容纳检索文档、对话历史和当前查询。

#### 9.2.6.1 滑动窗口策略

```
+======================================================================+
|                    多轮对话上下文滑动窗口                              |
+======================================================================+
|                                                                        |
|   对话轮次:  1    2    3    4    5    6    7    8    9    10          |
|             |    |    |    |    |    |    |    |    |    |             |
|             v    v    v    v    v    v    v    v    v    v             |
|                                                                        |
|   策略A: 固定窗口 (最后N轮)                                            |
|   +-------------------------------------------------------+           |
|   | 轮次8  | 轮次9  | 轮次10 | 系统提示 | 检索上下文 | Q  |           |
|   +-------------------------------------------------------+           |
|   优点：简单、Token可控  缺点：丢失早期上下文                      |
|                                                                        |
|   策略B: 摘要压缩窗口                                                  |
|   +-------------------------------------------------------+           |
|   | 早期摘要 | 轮次8 | 轮次9 | 轮次10 | 检索上下文 | Q |           |
|   +-------------------------------------------------------+           |
|   优点：保留历史脉络  缺点：摘要可能丢失细节                      |
|                                                                        |
|   策略C: 层次化窗口                                                    |
|   +-------------------------------------------------------+           |
|   | 核心记忆 | 近期详情 | 检索上下文 | 当前Q |            |           |
|   +-------------------------------------------------------+           |
|   优点：兼顾长期记忆和短期细节  缺点：实现复杂                    |
|                                                                        |
+======================================================================+
```

#### 9.2.6.2 对话历史管理实现

```python
@dataclass
class ConversationTurn:
    """单轮对话"""
    role: str  # 'user' 或 'assistant'
    content: str
    timestamp: float = 0.0
    metadata: Dict = field(default_factory=dict)


class ConversationContextManager:
    """多轮对话上下文管理器"""
    
    def __init__(self, 
                 max_history_tokens: int = 2000,
                 max_turns: int = 10,
                 summarization_model = None):
        self.max_history_tokens = max_history_tokens
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []
        self.summary: str = ""
        self.summarization_model = summarization_model
    
    def add_turn(self, turn: ConversationTurn):
        """添加新的对话轮次"""
        self.turns.append(turn)
        
        # 超过最大轮次数时触发压缩
        if len(self.turns) > self.max_turns:
            self._compress_early_turns()
    
    def _compress_early_turns(self):
        """将早期对话压缩为摘要"""
        if len(self.turns) <= 3:
            return
        
        # 保留最近3轮，压缩早期轮次
        early_turns = self.turns[:-3]
        recent_turns = self.turns[-3:]
        
        # 生成早期对话摘要
        early_text = '\n'.join([
            f"{t.role}: {t.content[:200]}" 
            for t in early_turns
        ])
        
        # 实际场景中调用摘要模型
        self.summary = self._generate_summary(early_text)
        self.turns = recent_turns
    
    def _generate_summary(self, text: str) -> str:
        """生成对话摘要（简化实现）"""
        # 实际应使用LLM生成结构化摘要
        key_points = []
        for line in text.split('\n')[:5]:
            if len(line) > 20:
                key_points.append(line[:100])
        return f"[对话摘要] 已讨论的主题: {'; '.join(key_points)}"
    
    def build_history_context(self) -> str:
        """构建对话历史上下文"""
        parts = []
        
        if self.summary:
            parts.append(f"[先前对话摘要]\n{self.summary}\n")
        
        if self.turns:
            parts.append("[最近对话]")
            for turn in self.turns:
                role_label = "用户" if turn.role == "user" else "助手"
                parts.append(f"{role_label}: {turn.content}")
        
        return '\n'.join(parts)
    
    def get_token_usage(self, token_counter) -> int:
        """获取对话历史的Token使用量"""
        return token_counter(self.build_history_context())
```

---

### 9.2.7 上下文窗口利用率优化

#### 9.2.7.1 优化策略全景

```
+======================================================================+
|                   上下文窗口利用率优化策略                             |
+======================================================================+
|                                                                        |
|  ┌─────────────────────────────────────────────────────────────────┐  |
|  │ 策略1: 自适应Token分配                                          │  |
|  │ 根据查询复杂度动态调整各部分比例:                               │  |
|  │   - 简单查询: 增大输出预留, 减少检索上下文                      │  |
|  │   - 复杂查询: 增大检索上下文, 减少输出预留                      │  |
|  │   - 多轮对话: 增大对话历史占比                                  │  |
|  └─────────────────────────────────────────────────────────────────┘  |
|                                                                        |
|  ┌─────────────────────────────────────────────────────────────────┐  |
|  │ 策略2: 内容去重与冗余消除                                       │  |
|  │ - 基于MinHash的文档指纹去重                                     │  |
|  │ - 基于Rouge-L的段落级内容重叠检测                               │  |
|  │ - 跨文档信息融合（相同实体、事件的信息合并）                   │  |
|  └─────────────────────────────────────────────────────────────────┘  |
|                                                                        |
|  ┌─────────────────────────────────────────────────────────────────┐  |
|  │ 策略3: 分块粒度动态调整                                         │  |
|  │ - 根据查询类型选择最优分块大小                                  │  |
|  │ - 事实查询: 小分块(128-256 tokens)                              │  |
|  │ - 分析查询: 大分块(512-1024 tokens)                              │  |
|  │ - 长文档查询: 混合粒度                                          │  |
|  └─────────────────────────────────────────────────────────────────┘  |
|                                                                        |
+======================================================================+
```

---

## 9.3 提示工程 (Prompt Engineering for RAG)

### 9.3.1 概念定义

**定义**：RAG系统中的提示工程是指设计、构造和优化输入给大语言模型的提示模板（Prompt Template），以引导模型正确理解上下文信息、遵循指令约束并生成高质量答案的系统工程方法。

**背景**：传统对话系统中的提示工程主要关注角色设定和任务描述。RAG系统引入检索上下文后，提示工程面临新的挑战：如何让模型区分"自身知识"和"检索到的知识"、如何约束模型仅依据检索内容回答、如何让模型正确引用信息来源。

**核心目标**：
- **忠实性** (Faithfulness)：答案必须准确反映检索到的上下文信息
- **可控性** (Controllability)：答案生成受限于指定的知识和规则
- **可追溯性** (Traceability)：每一条论断都可以追溯到具体的来源文档
- **鲁棒性** (Robustness)：面对噪声上下文仍能保持输出质量

---

### 9.3.2 提示模板设计模式

#### 9.3.2.1 标准RAG提示模板架构

```
+======================================================================+
|                    RAG 系统提示模板标准化架构                         |
+======================================================================+
|                                                                        |
|  ┌──────────────────────────────────────────────┐                     |
|  │         角色定义 (Role Definition)            │                     |
|  │  "你是一个专业的企业知识库助手..."             │                     |
|  │  定义: 身份、能力边界、语气风格               │                     |
|  ├──────────────────────────────────────────────┤                     |
|  │         行为约束 (Behavior Constraints)       │                     |
|  │  "仅根据提供的参考文档回答问题..."            │                     |
|  │  定义: 知识边界、引用要求、不回答的情况       │                     |
|  ├──────────────────────────────────────────────┤                     |
|  │         输出规范 (Output Specification)        │                     |
|  │  "回答格式: 1. 直接答案 2. 详细解释..."       │                     |
|  │  定义: 格式要求、结构要求、语言要求           │                     |
|  ├──────────────────────────────────────────────┤                     |
|  │         检索上下文 (Retrieved Context)         │                     |
|  │  [文档1]...[文档2]...[文档3]...               │                     |
|  │  动态注入：由检索系统实时提供的参考文档       │                     |
|  ├──────────────────────────────────────────────┤                     |
|  │         用户查询 (User Query)                  │                     |
|  │  用户的当前问题和相关约束                     │                     |
|  └──────────────────────────────────────────────┘                     |
|                                                                        |
+======================================================================+
```

#### 9.3.2.2 标准RAG提示模板实现

```python
class RAGPromptTemplate:
    """
    RAG提示模板引擎
    
    支持多种模板模式：
    1. 基础模板：系统提示 + 上下文 + 查询
    2. 严格溯源模板：强制引用
    3. 分析型模板：深度推理
    """
    
    # 基础模板
    BASIC_TEMPLATE = """
你是一个专业的知识库助手，名为{assistant_name}。
你的职责是根据提供的参考文档回答用户问题。

## 行为准则：
1. 优先使用参考文档中的信息回答问题
2. 如果参考文档中没有相关信息，请明确说明"根据现有资料，无法回答该问题"
3. 不要编造或假设文档中没有的信息
4. 回答时请引用具体的文档来源

## 输出格式：
- 先给出简洁的答案摘要
- 再提供详细解释，引用相关文档
- 最后列出信息来源

{context_section}

## 用户问题：
{query}

请回答：
""".strip()
    
    # 严格溯源模板
    STRICT_GROUNDING_TEMPLATE = """
你是一个严格的企业知识库助手。你必须**仅根据**以下提供的参考文档片段回答问题。

## 强制规则：
1. 【知识边界】你只能使用下方<documents>标签内的信息。如果问题涉及的
   信息不在文档中，必须回答"根据提供的资料，我无法回答此问题"。
2. 【禁止推测】严禁使用你的预训练知识或常识进行补充。不得进行任何
   形式的推测或假设。
3. 【引用要求】每个事实性断言必须附带引用标记 [文档X]，其中X是
   文档编号。
4. 【诚实原则】当文档信息相互矛盾时，请指出矛盾所在而非自行判断。
5. 【范围限定】当问题超出文档覆盖范围时，明确告知用户文档覆盖的
   知识边界。

## 参考文档：
<documents>
{context_section}
</documents>

## 用户问题：
{query}

请严格按照上述规则回答。每个事实陈述后必须标注文档来源。
""".strip()
    
    # 分析推理模板（Chain-of-Thought + RAG）
    COT_RAG_TEMPLATE = """
你是一个具备深度分析能力的知识库助手。请按照以下步骤分析并回答问题。

## 分析流程：
1. 【信息提取】从参考文档中提取与问题相关的关键信息片段
2. 【逻辑推理】对提取的信息进行逻辑分析和推理
3. 【结论生成】基于推理过程生成最终答案
4. 【来源追溯】为每一步推理标注信息来源

## 参考文档：
{context_section}

## 用户问题：
{query}

请逐步进行思考和分析：
""".strip()
    
    def render(self,
               template_type: str,
               context: str,
               query: str,
               assistant_name: str = "知识助手",
               extra_vars: Dict = None) -> str:
        """渲染提示模板"""
        templates = {
            "basic": self.BASIC_TEMPLATE,
            "strict": self.STRICT_GROUNDING_TEMPLATE,
            "cot": self.COT_RAG_TEMPLATE,
        }
        
        template = templates.get(template_type, self.BASIC_TEMPLATE)
        vars_dict = {
            "assistant_name": assistant_name,
            "context_section": context,
            "query": query,
            **(extra_vars or {})
        }
        
        return template.format(**vars_dict)
```

---

### 9.3.3 Few-Shot提示在RAG中的应用

在RAG上下文中，Few-Shot示例选择需要与当前查询和检索到的文档类型匹配。

```
+======================================================================+
|                RAG场景下的Few-Shot示例选择策略                        |
+======================================================================+
|                                                                        |
|  策略A: 查询相似度选择                                                 |
|  ┌────────────────────────────────────────────────────────┐           |
|  │ 当前查询 → Embedding → 与示例库中的query作语义匹配      │           |
|  │ → 选择Top-K最相似查询对应的示例对                       │           |
|  │ 优点: 考虑查询意图相似性                                │           |
|  │ 缺点: 未考虑检索上下文的结构差异                        │           |
|  └────────────────────────────────────────────────────────┘           |
|                                                                        |
|  策略B: 上下文结构匹配                                                 |
|  ┌────────────────────────────────────────────────────────┐           |
|  │ 当前检索结果 → 分析文档类型/数量/相关度分布             │           |
|  │ → 选择具有相似上下文结构的示例                          │           |
|  │ 优点: 示例与当前上下文情况匹配度高                      │           |
|  │ 缺点: 需要维护结构化的示例库                            │           |
|  └────────────────────────────────────────────────────────┘           |
|                                                                        |
|  策略C: 混合选择                                                       |
|  ┌────────────────────────────────────────────────────────┐           |
|  │ 结合查询语义相似度(60%) + 上下文结构相似度(40%)         │           |
|  │ → 加权评分 → 选择Top-K                                  │           |
|  │ 优点: 综合两种维度的匹配                                │           |
|  │ 缺点: 需要调优权重                                      │           |
|  └────────────────────────────────────────────────────────┘           |
|                                                                        |
+======================================================================+
```

```python
class FewShotSelector:
    """RAG场景下的Few-Shot示例选择器"""
    
    def __init__(self, example_store: List[Dict], embedding_model):
        self.example_store = example_store
        self.embedding_model = embedding_model
    
    def select_by_query_similarity(self, query: str, k: int = 3):
        """基于查询语义相似度选择示例"""
        query_emb = self.embedding_model.encode(query)
        
        scored = []
        for ex in self.example_store:
            ex_emb = self.embedding_model.encode(ex["query"])
            similarity = cosine_similarity([query_emb], [ex_emb])[0][0]
            scored.append((ex, similarity))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ex for ex, _ in scored[:k]]
    
    def build_few_shot_prompt(self, query: str, context: str, 
                              k: int = 2) -> str:
        """构建包含Few-Shot示例的提示"""
        examples = self.select_by_query_similarity(query, k)
        
        parts = ["以下是与当前问题类似的问答示例：\n"]
        for i, ex in enumerate(examples):
            parts.append(f"### 示例 {i+1}")
            parts.append(f"**背景信息**: {ex['context'][:200]}...")
            parts.append(f"**问题**: {ex['query']}")
            parts.append(f"**回答**: {ex['answer']}")
            parts.append("")
        
        parts.append("---\n现在请根据以下参考文档回答新问题：\n")
        return '\n'.join(parts)
```

---

### 9.3.4 来源引用与溯源技术 (Citation & Source Attribution)

来源引用是RAG系统可靠性的关键保障，特别是对于企业级应用（法律、金融、医疗等场景）。

#### 9.3.4.1 引用格式对比

| 引用格式 | 示例 | 优点 | 缺点 | 适用场景 |
|---------|------|------|------|---------|
| 内联引用 | "...这是结论[1](source.pdf p.3)" | 精确、可验证 | 影响阅读流畅性 | 学术、法律 |
| 脚注式 | "...这是结论^(1)\n\n---\n1. source.pdf, p.3" | 阅读流畅 | 需额外解析 | 报告、文章 |
| 结构化引用 | `{"claim":"...", "source":"doc_1", "page":3}` | 机器可解析 | 不适合直接展示 | API、结构化输出 |
| 段落标注 | "参考来源: [文档3] 第2段 [文档5] 第1段" | 简洁直观 | 粒度较粗 | 一般问答 |

#### 9.3.4.2 引用提取实现

```python
import re
import json
from typing import List, Tuple

class CitationExtractor:
    """从LLM输出中提取和管理引用信息"""
    
    INLINE_PATTERN = re.compile(r'\[(\d+)\]|\[文档(\d+)\]|\[来源(\d+)\]')
    STRUCTURED_PATTERN = re.compile(r'<cite\s+doc="([^"]+)"(?:\s+page="([^"]+)")?\s*/>')
    
    @classmethod
    def extract_inline_citations(cls, text: str) -> List[Tuple[int, str]]:
        """提取内联引用，返回 (位置, 文档编号) 列表"""
        citations = []
        for match in cls.INLINE_PATTERN.finditer(text):
            doc_num = match.group(1) or match.group(2) or match.group(3)
            citations.append((match.start(), doc_num))
        return citations
    
    @classmethod
    def extract_structured_citations(cls, text: str) -> List[Dict]:
        """提取结构化引用"""
        citations = []
        for match in cls.STRUCTURED_PATTERN.finditer(text):
            citations.append({
                "document_id": match.group(1),
                "page": match.group(2),
                "position": match.start()
            })
        return citations
    
    @classmethod
    def verify_citations(cls, 
                         answer: str, 
                         source_documents: List[Dict]) -> Dict[str, Any]:
        """
        验证引用的完整性和准确性
        
        检查项：
        1. 所有引用编号是否已存在对应文档
        2. 重要断言是否有引用支持
        3. 引用的信息是否与源文档一致（需要人工或LLM辅助验证）
        """
        citations = cls.extract_inline_citations(answer)
        referenced_docs = set(int(num) for _, num in citations)
        available_docs = set(doc.get('index', 0) for doc in source_documents)
        
        missing_refs = referenced_docs - available_docs
        unused_docs = available_docs - referenced_docs
        
        return {
            "total_citations": len(citations),
            "unique_docs_cited": len(referenced_docs),
            "total_available_docs": len(available_docs),
            "missing_references": list(missing_refs),
            "unused_documents": list(unused_docs),
            "citation_coverage": len(referenced_docs) / max(1, len(available_docs)),
            "is_valid": len(missing_refs) == 0
        }


class SourceAttributor:
    """
    来源属性标注器
    
    在构建的提示中强制要求模型进行来源标注，
    并在输出后验证引用质量。
    """
    
    @staticmethod
    def build_citation_instruction(style: str = "inline") -> str:
        """生成引用指令"""
        instructions = {
            "inline": """
## 引用格式要求：
- 每一条事实性信息后必须标注其来源文档编号，格式为 [文档X]
- 如果一条信息来自多个文档，标注所有来源：如 [文档1, 文档3]
- 如果信息是你的推理而非文档内容，标注为 [推理]
- 示例："该公司2023年营收增长了15% [文档2][文档5]"
""",
            "footnote": """
## 引用格式要求：
- 在文本中使用上标数字标记引用：标记为 ^1^, ^2^ 等
- 在回答末尾列出所有参考文献
- 格式示例："营收增长15%^1^
...
参考文献：
1. 2023年度财务报告，第3页
2. 行业分析报告，第12页
"""
        }
        return instructions.get(style, instructions["inline"])
```

---

### 9.3.5 来源锚定 (Source Grounding)

来源锚定是RAG系统反幻觉的核心技术，强制LLM仅基于提供的上下文生成答案。

```python
class SourceGroundingPrompt:
    """
    来源锚定提示构建器
    
    通过多层次约束，确保LLM输出严格基于检索上下文：
    1. 明确的知识边界
    2. 强制的引用要求
    3. 结构化的输出格式
    4. 自我检查机制
    """
    
    GROUNDING_SYSTEM_PROMPT = """
## 身份
你是一个严格的知识库助手。你的全部知识来源于下方<knowledge_base>标签中的文档。

## 核心规则 - 必须严格遵守：
1. **仅使用提供的文档**: 你只能使用<knowledge_base>中的信息回答。
   不要使用你的训练数据中的任何外部知识。
2. **不知即言不知**: 如果提供的文档不包含回答问题所需的信息，
   直接说"根据现有资料，我无法回答此问题"，不要编造答案。
3. **强制引用**: 你输出的每一句事实性陈述，都必须标注其来源文档编号。
4. **矛盾处理**: 如果不同文档提供了相互矛盾的信息，列出所有观点
   及其来源，由用户判断。
5. **范围意识**: 注意文档的覆盖范围。如果问题部分超出范围，
   只回答文档覆盖的部分，并说明局限性。

## 引用格式：
每个事实后标注来源，格式: [来源: 文档X]
如果基于多个文档: [来源: 文档X, 文档Y]
如果是你的推理: [推理]

## 质量检查：
在回答末尾，请进行自我检查：
- [ ] 所有事实是否都有来源标注？
- [ ] 有没有使用文档外的信息？
- [ ] 矛盾信息是否已被指出？
"""
    
    @classmethod
    def build_grounded_prompt(cls, context: str, query: str) -> str:
        return f"""{cls.GROUNDING_SYSTEM_PROMPT}

<knowledge_base>
{context}
</knowledge_base>

## 用户问题
{query}

请基于以上知识库，严格遵循规则回答。"""
```

---

### 9.3.6 反幻觉提示 (Anti-Hallucination Prompts)

幻觉（Hallucination）是RAG系统最严峻的挑战之一。反幻觉提示通过多层次约束来抑制模型的编造行为。

```
+======================================================================+
|                      反幻觉提示设计层次结构                           |
+======================================================================+
|                                                                        |
|  第一层: 知识边界 (Knowledge Boundary)                                 |
|  ┌────────────────────────────────────────────────────────────────┐   |
|  │ "你只能使用以下提供的参考文档中的信息来回答..."                  │   |
|  │ "如果文档不包含相关信息，请明确声明'我不知道'..."               │   |
|  └────────────────────────────────────────────────────────────────┘   |
|                                                                        |
|  第二层: 事实性约束 (Factual Constraint)                                |
|  ┌────────────────────────────────────────────────────────────────┐   |
|  │ "对于每个断言，请给出明确的来源标注..."                          │   |
|  │ "如果不同的文档给出矛盾信息，请指出矛盾..."                     │   |
|  └────────────────────────────────────────────────────────────────┘   |
|                                                                        |
|  第三层: 不确定性表达 (Uncertainty Expression)                          |
|  ┌────────────────────────────────────────────────────────────────┐   |
|  │ "使用适当的确定性级别: 确定性/可能性/推测..."                    │   |
|  │ "当信息不完整时，说明信息的局限性..."                           │   |
|  └────────────────────────────────────────────────────────────────┘   |
|                                                                        |
|  第四层: 自我校验 (Self-Verification)                                   |
|  ┌────────────────────────────────────────────────────────────────┐   |
|  │ "在回答末尾，检查你的回答是否完全基于提供的文档..."              │   |
|  │ "标记任何不确定性或需要进一步验证的信息..."                     │   |
|  └────────────────────────────────────────────────────────────────┘   |
|                                                                        |
+======================================================================+
```

```python
ANTI_HALLUCINATION_PROMPT = """
## 反幻觉强制指令

你正在提供一个**企业级知识库服务**。准确性是最高的优先级。

### 知识使用规则：
1. **封闭知识域**: 你的全部知识在当前对话中仅限于下面<reference>标签中的内容。
   忽略你从训练数据中获得的所有其他知识。
2. **不知即言不知**: 如果<reference>中没有包含答案所需的信息，请直接声明：
   "根据提供的信息，我无法回答该问题。可能的原因: [列举可能的信息缺失]"
3. **不确定度标注**: 对于每个主要结论，使用以下标签标注确定度：
   - [确定] - 信息在文档中明确陈述，且来源可靠
   - [可能] - 信息需要从文档中推理，或只有部分支持
   - [存疑] - 多个文档信息存在分歧
4. **禁止行为**: 
   - 禁止编造文档中不存在的数据、日期、名称或统计信息
   - 禁止对文档信息进行未经证实的扩展推断
   - 禁止将常识或世界知识作为事实来源（除非在推理中有明确标注）

### 输出格式：
1. **答案摘要**: 用1-2句话总结核心答案
2. **详细分析**: 基于文档的详细分析，每句标注来源和确定度
3. **信息缺口**: 列出文档中缺失但问题相关的重要信息
4. **自我审核**: 确认所有陈述都有文档依据

---
{context}
---

问题: {query}

请严格按照反幻觉规则回答：
"""
```

---

### 9.3.7 Chain-of-Thought + RAG集成

将思维链推理与RAG系统结合，可以显著提升复杂推理问题的回答质量。

```
+======================================================================+
|                   CoT+RAG 集成推理流程                                |
+======================================================================+
|                                                                        |
|  用户查询: "对比分析产品A和产品B在2023年的市场表现"                   |
|                                                                        |
|  Step 1: 查询分解                                                      |
|  ┌──────────────────────────────────────────────────────┐            |
|  │ 子查询1: "产品A 2023年 市场份额"                      │            |
|  │ 子查询2: "产品B 2023年 市场份额"                      │            |
|  │ 子查询3: "产品A 2023年 营收增长率"                    │            |
|  │ 子查询4: "产品B 2023年 营收增长率"                    │            |
|  │ 子查询5: "产品A vs 产品B 竞争分析 2023"               │            |
|  └──────────────────────────────────────────────────────┘            |
|                          |                                              |
|                          v                                              |
|  Step 2: 并行检索 + 上下文注入                                         |
|  ┌──────────────────────────────────────────────────────┐            |
|  │ 检索系统为每个子查询返回相关文档 → 合并去重打包         │            |
|  └──────────────────────────────────────────────────────┘            |
|                          |                                              |
|                          v                                              |
|  Step 3: 逐步推理 (Chain-of-Thought)                                   |
|  ┌──────────────────────────────────────────────────────┐            |
|  │ 思考: 首先提取产品A的2023年市场份额...  [文档1][文档2] │            |
|  │ 思考: 然后提取产品B的2023年市场份额...  [文档3][文档4] │            |
|  │ 思考: 对比两者的市场份额变化趋势...                    │            |
|  │ 思考: 分析营收数据...                                  │            |
|  │ 思考: 综合以上分析，得出结论...                        │            |
|  └──────────────────────────────────────────────────────┘            |
|                          |                                              |
|                          v                                              |
|  Step 4: 答案生成                                                      |
|  ┌──────────────────────────────────────────────────────┐            |
|  │ 结构化对比分析答案 + 完整引用链                         │            |
|  └──────────────────────────────────────────────────────┘            |
|                                                                        |
+======================================================================+
```

```python
class CoTRAGPipeline:
    """
    Chain-of-Thought + RAG 集成管道
    
    实现查询分解、并行检索、逐步推理和结构化输出的完整流程
    """
    
    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm_client = llm_client
    
    def decompose_query(self, query: str) -> List[str]:
        """将复杂查询分解为子查询"""
        decomposition_prompt = f"""将以下复杂问题分解为2-5个简单的子问题，
每个子问题应该可以独立从文档中检索答案。

复杂问题: {query}

请输出子问题列表（每行一个，以"- "开头）："""
        
        response = self.llm_client.generate(decomposition_prompt)
        # 解析子问题
        sub_queries = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('- ') or line.startswith('- '):
                sub_queries.append(line[2:])
        return sub_queries[:5]  # 最多5个子问题
    
    def execute_cot_rag(self, query: str, max_sub_queries: int = 5) -> Dict:
        """执行CoT+RAG完整流程"""
        
        # Step 1: 查询分解
        sub_queries = self.decompose_query(query)
        sub_queries = sub_queries[:max_sub_queries]
        
        # Step 2: 并行检索
        all_docs = []
        for sub_q in sub_queries:
            docs = self.retriever.retrieve(sub_q, top_k=3)
            all_docs.extend(docs)
        
        # 去重并按相关性排序
        unique_docs = self._deduplicate_and_rank(all_docs)
        
        # Step 3: 构建CoT上下文
        context = ContextPacker(token_counter=lambda x: len(x)//2,
                                max_context_tokens=6000).pack(
            unique_docs, query, format_type="xml"
        )
        
        # Step 4: CoT推理提示
        cot_prompt = f"""请按照思维链方式逐步分析以下问题。

## 分析步骤：
1. 首先，从参考文档中提取与问题相关的所有关键信息点
2. 然后，对提取的信息进行逻辑分析和推理
3. 接着，考虑不同文档之间的信息是否一致
4. 最后，得出综合性的结论

## 参考文档：
{context}

## 需要回答的问题：
{query}

## 请按照以下格式逐步思考：

**步骤1 - 信息提取：**
（列出从各文档中提取的关键信息，标注来源）

**步骤2 - 逻辑分析：**
（对信息进行分析、比较和推理）

**步骤3 - 一致性检查：**
（检查不同来源的信息是否一致，如有矛盾请指出）

**步骤4 - 最终结论：**
（给出综合性的最终答案）"""
        
        answer = self.llm_client.generate(cot_prompt)
        
        return {
            "query": query,
            "sub_queries": sub_queries,
            "retrieved_docs": len(unique_docs),
            "answer": answer,
            "citations": CitationExtractor.extract_inline_citations(answer)
        }
    
    def _deduplicate_and_rank(self, docs: List) -> List:
        """去重并排序"""
        seen = set()
        unique = []
        for doc in sorted(docs, key=lambda x: x.relevance_score, reverse=True):
            if doc.doc_id not in seen:
                seen.add(doc.doc_id)
                unique.append(doc)
        return unique
```

---

## 9.4 关键技术挑战

### 9.4.1 上下文长度限制详解

上下文窗口是RAG系统的基础性约束。不同窗口大小的模型需要不同的上下文构建策略。

```
+==========================================================================+
|                    上下文窗口大小演进与技术影响                           |
+==========================================================================+
|                                                                           |
|  代际    窗口大小     代表模型                 核心挑战                   |
|  ────── ──────────── ────────────────────── ────────────────────────     |
|  第一代   4K         GPT-3.5, LLaMA-1       Token预算极度有限             |
|                                              需要严格的上下文压缩        |
|                                                                           |
|  第二代   8K-32K     GPT-4(8K), Mistral-7B   可容纳10-50个文档片段       |
|                                              Lost in Middle问题出现      |
|                                                                           |
|  第三代   128K-200K  GPT-4-Turbo, Claude-3   长文档直接处理可行           |
|                                              但推理延时显著增加          |
|                                                                           |
|  第四代   1M+        Gemini-1.5-Pro          几乎无文档长度限制           |
|                                              但注意力稀释问题加剧        |
|                                                                           |
+==========================================================================+
|                                                                           |
|  关键发现：                                                               |
|  ┌─────────────────────────────────────────────────────────────────┐     |
|  │ 1. 更大的窗口不等于更好的性能                                     │     |
|  │    - 在128K上下文中，如果关键信息在中间位置，模型可能"忽略"它    │     |
|  │    - "Needle in a Haystack"测试：信息位置是准确率的关键变量      │     |
|  │                                                                   │     |
|  │ 2. 上下文越长，推理速度越慢、成本越高                             │     |
|  │    - 4K上下文: ~1秒, 128K: ~20-60秒                              │     |
|  │    - API成本与输入Token数成正比                                   │     |
|  │                                                                   │     |
|  │ 3. 最优上下文大小因查询类型而异                                   │     |
|  │    - 事实查询: 2K-4K 足够                                        │     |
|  │    - 分析查询: 8K-32K 合适                                        │     |
|  │    - 综合报告: 32K-64K 可能必要                                   │     |
|  └─────────────────────────────────────────────────────────────────┘     |
|                                                                           |
+==========================================================================+
```

**各窗口大小对应的RAG策略建议**：

| 窗口大小 | 建议块数 | 每块Token | 压缩策略 | 排序策略 |
|---------|---------|----------|---------|---------|
| 4K | 3-5 | 300-500 | 高度压缩 | 相关性降序 |
| 8K | 5-10 | 300-500 | 中度压缩 | 分层排序 |
| 32K | 10-20 | 500-800 | 轻度压缩 | 分层排序 |
| 128K | 20-50 | 500-1000 | 可选项 | Lost-in-Middle优化 |
| 1M | 50-200 | 500-2000 | 不压缩 | 分层+时间混合 |

---

### 9.4.2 提示污染防护 (Prompt Pollution Prevention)

提示污染是指无关、低质量或相互矛盾的检索结果混入上下文，导致模型输出质量下降的现象。

```
+======================================================================+
|                      提示污染的来源与防护                              |
+======================================================================+
|                                                                        |
|  污染来源:                                                              |
|  ┌──────────────────────────────────────────────────────────────┐    |
|  │ 1. 主题无关文档: 检索返回了与问题主题相似但内容无关的文档    │    |
|  │ 2. 低质量内容: 包含错误信息、过时信息的文档                  │    |
|  │ 3. 矛盾信息: 多个文档对同一事实给出不同甚至相反的描述        │    |
|  │ 4. 重复信息: 相同内容的不同副本占用上下文空间                │    |
|  │ 5. 不完整片段: 检索到的Chunk缺少必要的上下文背景             │    |
|  └──────────────────────────────────────────────────────────────┘    |
|                                                                        |
|  防护策略:                                                              |
|  ┌──────────────────────────────────────────────────────────────┐    |
|  │ [入口防护]                                                     │    |
|  │  - 检索后重排序: 使用Cross-Encoder提高相关性判断精度          │    |
|  │  - 相关性阈值过滤: score < threshold 的文档直接丢弃           │    |
|  │  - 多样性过滤: MMR算法平衡相关性和多样性                      │    |
|  │                                                                 |    |
|  │ [内容防护]                                                     │    |
|  │  - 文档质量评分: 基于来源权威性、内容完整性、时效性评分       │    |
|  │  - 矛盾检测: 使用NLI模型检测文档间的事实矛盾                  │    |
|  │  - 信息一致性校验: 交叉验证关键信息在多个来源中的一致性       │    |
|  │                                                                 |    |
|  │ [生成防护]                                                     │    |
|  │  - 严格来源锚定: 约束模型仅使用文档信息                       │    |
|  │  - 不确定性表达: 要求模型标注信息的确定度                     │    |
|  │  - 后生成验证: 使用NLI模型验证答案与源文档的一致性            │    |
|  └──────────────────────────────────────────────────────────────┘    |
|                                                                        |
+======================================================================+
```

#### 9.4.2.1 上下文冗余检测实现

```python
class RedundancyDetector:
    """
    上下文冗余检测器
    
    检测并移除上下文中的冗余信息，提升信息密度
    """
    
    def __init__(self, similarity_threshold: float = 0.80):
        self.threshold = similarity_threshold
    
    def detect_semantic_redundancy(self, 
                                    chunks: List[str]) -> Dict[int, List[int]]:
        """
        基于语义相似度的冗余检测
        
        返回: {chunk_index: [冗余的chunk索引列表]}
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        if len(chunks) <= 1:
            return {}
        
        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(chunks)
        
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(tfidf_matrix)
        
        redundant_pairs = {}
        for i in range(len(chunks)):
            redundant_pairs[i] = []
            for j in range(i + 1, len(chunks)):
                if sim_matrix[i][j] >= self.threshold:
                    redundant_pairs[i].append(j)
        
        return redundant_pairs
    
    def detect_factual_contradiction(self, 
                                      claim1: str, 
                                      claim2: str) -> float:
        """
        检测两个声明之间的事实矛盾
        
        使用NLI (Natural Language Inference) 模型检测
        返回矛盾概率 (0-1)
        """
        # 实际实现中使用NLI模型如RoBERTa-MNLI
        # 这里展示调用模式
        prompt = f"""判断以下两个陈述之间的关系：
陈述A: {claim1}
陈述B: {claim2}

关系判断（选择: 等价/相关/无关/矛盾）：
如果矛盾，说明矛盾点："""
        
        # response = self.nli_model.predict(claim1, claim2)
        # return response.contradiction_probability
        return 0.0  # 占位
    
    def remove_redundant(self, 
                         chunks: List[str], 
                         keep_strategy: str = "first") -> List[str]:
        """
        移除冗余chunk
        
        keep_strategy:
        - "first": 保留第一个出现的chunk
        - "longest": 保留内容最长的chunk
        - "highest_quality": 保留质量最高的chunk
        """
        redundant_map = self.detect_semantic_redundancy(chunks)
        
        to_remove = set()
        for idx, redundant_indices in redundant_map.items():
            if idx not in to_remove:
                to_remove.update(redundant_indices)
        
        return [c for i, c in enumerate(chunks) if i not in to_remove]
```

#### 9.4.2.2 最优上下文大小指引

```python
class ContextSizeOptimizer:
    """
    根据查询类型推荐最优上下文大小
    """
    
    QUERY_TYPE_GUIDELINES = {
        "factual": {
            "description": "事实型查询 (如'X的定义是什么', 'Y事件发生在何时')",
            "optimal_docs": 3,
            "optimal_tokens": "1K-2K",
            "chunk_size": "128-256 tokens",
            "strategy": "精选取前3个高相关文档"
        },
        "analytical": {
            "description": "分析型查询 (如'分析X的原因', '对比A和B的差异')",
            "optimal_docs": 5,
            "optimal_tokens": "4K-8K",
            "chunk_size": "256-512 tokens",
            "strategy": "分层排序，确保多角度覆盖"
        },
        "synthesis": {
            "description": "综合型查询 (如'撰写X主题的综述报告')",
            "optimal_docs": 12,
            "optimal_tokens": "16K-32K",
            "chunk_size": "512-1024 tokens",
            "strategy": "使用摘要压缩，保留信息广度"
        },
        "creative": {
            "description": "创意型查询 (如'基于X设计Y方案')",
            "optimal_docs": 4,
            "optimal_tokens": "2K-4K",
            "chunk_size": "256-512 tokens",
            "strategy": "提供启发性材料而非限制性材料"
        },
        "compliance": {
            "description": "合规型查询 (如'X规定是否允许Y')",
            "optimal_docs": "全部相关",
            "optimal_tokens": "8K-32K",
            "chunk_size": "原始片段",
            "strategy": "最小化压缩，保留完整上下文"
        }
    }
    
    @classmethod
    def recommend(cls, query_type: str) -> Dict:
        """根据查询类型推荐上下文大小配置"""
        return cls.QUERY_TYPE_GUIDELINES.get(
            query_type, 
            cls.QUERY_TYPE_GUIDELINES["analytical"]
        )
```

---

## 9.5 企业级实践

### 9.5.1 提示版本管理

在大型企业RAG系统中，提示模板是核心资产，需要专业的版本管理策略。

```
+======================================================================+
|                      提示版本管理系统架构                             |
+======================================================================+
|                                                                        |
|  ┌────────────────────────────────────────────────────────────┐       |
|  │                    Prompt Registry（提示注册中心）           │       |
|  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │       |
|  │  │ 开发环境  │  │ 测试环境  │  │ 预发环境  │  │ 生产环境  │   │       |
|  │  │ dev-v1.3  │  │ test-v1.2│  │ stage-v1 │  │ prod-v1  │   │       |
|  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │       |
|  └────────────────────────────────────────────────────────────┘       |
|                                                                        |
|  ┌────────────────────────────────────────────────────────────┐       |
|  │                    A/B Testing Engine                        │       |
|  │                                                              │       |
|  │   用户请求 ──→ 分流(50/50) ──→ 提示A ──→ 模型 ──→ 结果A    │       |
|  │                            ──→ 提示B ──→ 模型 ──→ 结果B    │       |
|  │                                                              │       |
|  │   评估指标:                                                   │       |
|  │   - 忠实度 (Faithfulness)                                    │       |
|  │   - 用户满意度 (User Satisfaction)                           │       |
|  │   - 引用准确率 (Citation Accuracy)                           │       |
|  │   - 平均Token消耗 (Avg Token Usage)                          │       |
|  └────────────────────────────────────────────────────────────┘       |
|                                                                        |
+======================================================================+
```

#### 9.5.1.1 提示注册中心实现

```python
import hashlib
import json
from datetime import datetime
from enum import Enum

class PromptStatus(Enum):
    DRAFT = "draft"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ROLLED_BACK = "rolled_back"

@dataclass
class PromptVersion:
    """提示模板版本"""
    id: str
    name: str
    version: str
    template: str
    variables: List[str]
    status: PromptStatus
    created_at: datetime
    updated_at: datetime
    author: str
    description: str
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    parent_version: Optional[str] = None


class PromptRegistry:
    """
    企业级提示注册中心
    
    功能：
    1. 提示模板的版本化存储
    2. 环境隔离（开发/测试/生产）
    3. 回滚与审计
    4. A/B测试支持
    """
    
    def __init__(self, storage_backend):
        self.storage = storage_backend
        self.active_ab_tests: Dict[str, Dict] = {}
    
    def register(self, prompt: PromptVersion) -> str:
        """注册一个提示版本"""
        prompt.updated_at = datetime.now()
        self.storage.save(prompt)
        return prompt.id
    
    def get_active_prompt(self, name: str, 
                          environment: str = "production") -> PromptVersion:
        """获取指定环境的当前活跃提示"""
        return self.storage.get_active(name, environment)
    
    def promote(self, prompt_id: str, 
                target_env: str) -> PromptVersion:
        """将提示从一个环境提升到另一个环境"""
        prompt = self.storage.get(prompt_id)
        
        env_order = ["draft", "testing", "staging", "production"]
        if env_order.index(target_env) <= env_order.index(prompt.status.value):
            raise ValueError(f"只能向前推进环境：当前{prompt.status}")
        
        # 归档旧版本
        old = self.storage.get_active(prompt.name, target_env)
        if old:
            old.status = PromptStatus.DEPRECATED
            self.storage.save(old)
        
        prompt.status = PromptStatus(target_env)
        self.storage.save(prompt)
        return prompt
    
    def rollback(self, name: str, 
                 environment: str = "production") -> PromptVersion:
        """回滚到上一个生产版本"""
        current = self.get_active_prompt(name, environment)
        previous = self.storage.get_previous_version(name, environment)
        
        if previous:
            current.status = PromptStatus.ROLLED_BACK
            self.storage.save(current)
            previous.status = PromptStatus(environment)
            self.storage.save(previous)
            return previous
        
        raise ValueError("没有可回滚的历史版本")
    
    def start_ab_test(self, 
                      name: str, 
                      variant_a_id: str, 
                      variant_b_id: str,
                      traffic_split: float = 0.5) -> str:
        """启动A/B测试"""
        test_id = f"ab_{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.active_ab_tests[test_id] = {
            "name": name,
            "variant_a": variant_a_id,
            "variant_b": variant_b_id,
            "traffic_split": traffic_split,
            "started_at": datetime.now(),
            "results": {"a": {}, "b": {}}
        }
        return test_id
    
    def end_ab_test(self, test_id: str) -> Dict:
        """结束A/B测试并返回统计结果"""
        test = self.active_ab_tests.pop(test_id, None)
        if not test:
            raise ValueError(f"测试 {test_id} 不存在")
        
        # 统计结果
        results_a = test["results"]["a"]
        results_b = test["results"]["b"]
        
        return {
            "test_id": test_id,
            "winner": "A" if self._is_significant_winner(results_a, results_b) 
                      else "B" if self._is_significant_winner(results_b, results_a)
                      else "无显著差异",
            "metrics_comparison": {
                "variant_a": results_a,
                "variant_b": results_b
            }
        }
    
    def _is_significant_winner(self, a: Dict, b: Dict) -> bool:
        """判断A是否显著优于B（简化实现）"""
        if not a or not b:
            return False
        a_score = a.get("composite_score", 0)
        b_score = b.get("composite_score", 0)
        return a_score > b_score * 1.05  # 5%显著性阈值
    
    def get_version_history(self, name: str) -> List[PromptVersion]:
        """获取提示的完整版本历史"""
        return self.storage.get_history(name)
```

---

### 9.5.2 域适配提示模板库

```python
class DomainPromptGallery:
    """
    领域特定的提示模板库
    
    企业建议维护按行业/领域分类的提示模板库，
    每个模板经过对应领域的质量验证。
    """
    
    TEMPLATES = {
        "legal": {
            "name": "法律咨询模板",
            "system_prompt": """
你是一个法律知识助手，专精于{jurisdiction}法律体系。
- 你的回答应基于提供的法律文件和判例
- 引用具体的法条编号和判例名称
- 区分"法律确定性结论"和"法律分析意见"
- 在不确定时明确建议咨询专业律师
""",
            "citation_style": "legal",  # 法条引用格式
            "chunk_size": 512,
        },
        "medical": {
            "name": "医疗知识模板",
            "system_prompt": """
你是一个医学知识参考助手。
- 你的回答仅供参考，不构成医疗建议
- 引用具体的医学文献和指南
- 区分"临床指南推荐"和"研究文献发现"
- 必须声明信息的时效性（如指南年份）
""",
            "citation_style": "inline",
            "chunk_size": 256,
        },
        "finance": {
            "name": "金融分析模板",
            "system_prompt": """
你是一个金融信息分析助手。
- 基于提供的金融数据和报告进行分析
- 区分"数据事实"和"分析推断"
- 标注数据的时间节点和口径
- 说明分析方法的局限性
""",
            "citation_style": "structured",
            "chunk_size": 512,
        },
        "technical": {
            "name": "技术文档模板",
            "system_prompt": """
你是一个技术文档助手。
- 基于提供的技术文档、代码和API参考回答问题
- 代码示例需要标明来源文件
- 区分"官方文档记载"和"社区最佳实践"
- 标注API版本和兼容性信息
""",
            "citation_style": "inline",
            "chunk_size": 1024,
        }
    }
    
    @classmethod
    def get_template(cls, domain: str) -> Dict:
        """获取领域模板"""
        return cls.TEMPLATES.get(domain, cls.TEMPLATES["technical"])
```

---

### 9.5.3 动态提示选择

基于查询分类自动选择最适合的提示模板和上下文构建策略。

```python
class DynamicPromptSelector:
    """
    动态提示选择器
    
    流程：
    用户查询 → 查询分类 → 选择提示模板 → 选择上下文策略 → 生成答案
    """
    
    def __init__(self, prompt_registry: PromptRegistry, 
                 query_classifier):
        self.registry = prompt_registry
        self.classifier = query_classifier
    
    def select(self, query: str, 
               retrieved_docs: List) -> Dict[str, Any]:
        """
        根据查询特征动态选择最佳提示策略
        
        返回：{
            "prompt_template": 选中的模板,
            "context_strategy": 上下文构建策略,
            "rationale": 选择理由
        }
        """
        # Step 1: 查询分类
        query_type = self.classifier.classify(query)
        
        # Step 2: 根据查询类型选择模板和策略
        strategy_map = {
            "factual": {
                "template": "basic",
                "order": "relevance_desc",
                "max_docs": 3,
                "format": "xml",
                "rationale": "事实查询需要精取高相关文档"
            },
            "analytical": {
                "template": "cot",
                "order": "hierarchical",
                "max_docs": 10,
                "format": "detailed",
                "rationale": "分析查询需要多角度覆盖和推理链"
            },
            "compliance": {
                "template": "strict",
                "order": "chronological",
                "max_docs": 20,
                "format": "detailed",
                "rationale": "合规查询需要完整保留法规上下文"
            },
            "creative": {
                "template": "basic",
                "order": "lost_in_middle",
                "max_docs": 5,
                "format": "minimal",
                "rationale": "创造性查询需要启发性而非限制性材料"
            }
        }
        
        strategy = strategy_map.get(query_type, strategy_map["analytical"])
        
        # Step 3: 获取模板
        template = self.registry.get_active_prompt(strategy["template"])
        
        return {
            "query_type": query_type,
            "prompt_template": template,
            "context_strategy": strategy,
            "rationale": strategy["rationale"]
        }
```

---

### 9.5.4 提示质量评估指标

```
+======================================================================+
|                     提示质量评估体系                                   |
+======================================================================+
|                                                                        |
|  一级指标                  二级指标                 评估方式           |
|  ─────────────────────── ────────────────────── ──────────────────     |
|                                                                        |
|  1. 忠实度 (Faithfulness)                                              |
|     ├─ 事实一致性           答案中的事实与源文档是否一致   NLI模型     |
|     ├─ 无幻觉率             不包含源文档中没有的信息      人工/LLM     |
|     └─ 引用准确率           引用指向的文档是否包含声称信息 自动比对     |
|                                                                        |
|  2. 相关性 (Relevance)                                                |
|     ├─ 答案相关性           答案与问题的匹配程度          LLM评分     |
|     ├─ 信息覆盖率           答案覆盖了检索文档中多少关键信息 BERTScore  |
|     └─ 冗余度               答案中不必要的重复信息比例    词汇分析     |
|                                                                        |
|  3. 完整性 (Completeness)                                              |
|     ├─ 关键信息包含率       问题所需的全部信息是否被覆盖  人工标注     |
|     ├─ 多角度覆盖           是否从不同角度回答了问题      语义分析     |
|     └─ 边界说明             是否说明了信息的局限性        LLM检测     |
|                                                                        |
|  4. 效率 (Efficiency)                                                  |
|     ├─ Token效率            信息密度(tokens per fact)     Token计数    |
|     ├─ 响应延迟             P50/P95/P99延迟              监控系统     |
|     └─ 检索利用率           被引用的检索文档比例          引用分析     |
|                                                                        |
+======================================================================+
```

```python
class PromptQualityEvaluator:
    """
    提示质量评估器
    
    自动化评估RAG输出的质量，支持多维度打分
    """
    
    def __init__(self, nli_model=None, embedding_model=None):
        self.nli_model = nli_model
        self.embedding_model = embedding_model
    
    def evaluate_faithfulness(self, 
                               answer: str, 
                               source_docs: List[str]) -> Dict:
        """
        评估答案忠实度
        
        核心方法：将答案拆分为原子声明，逐一与源文档比对
        """
        # 将答案拆分为原子声明
        claims = self._extract_atomic_claims(answer)
        
        supported = 0
        contradicted = 0
        unverifiable = 0
        
        for claim in claims:
            verdict = self._verify_claim_against_sources(claim, source_docs)
            if verdict == "supported":
                supported += 1
            elif verdict == "contradicted":
                contradicted += 1
            else:
                unverifiable += 1
        
        total = len(claims)
        return {
            "faithfulness_score": supported / max(1, total),
            "hallucination_rate": unverifiable / max(1, total),
            "contradiction_rate": contradicted / max(1, total),
            "total_claims": total,
            "supported_claims": supported,
            "contradicted_claims": contradicted,
            "unverifiable_claims": unverifiable
        }
    
    def evaluate_relevance(self, answer: str, query: str) -> float:
        """评估答案相关性"""
        if self.embedding_model:
            answer_emb = self.embedding_model.encode(answer)
            query_emb = self.embedding_model.encode(query)
            from sklearn.metrics.pairwise import cosine_similarity
            return float(cosine_similarity([answer_emb], [query_emb])[0][0])
        return 0.0
    
    def evaluate_completeness(self, 
                               answer: str, 
                               expected_key_points: List[str]) -> Dict:
        """
        评估答案完整性
        
        检查答案是否覆盖了所有预期的关键信息点
        """
        covered = 0
        for point in expected_key_points:
            # 简化的覆盖检测
            if point.lower() in answer.lower():
                covered += 1
        
        total = len(expected_key_points)
        return {
            "completeness_score": covered / max(1, total),
            "covered_points": covered,
            "total_expected_points": total,
            "missed_points": [
                p for p in expected_key_points 
                if p.lower() not in answer.lower()
            ]
        }
    
    def _extract_atomic_claims(self, text: str) -> List[str]:
        """将文本拆分为原子声明"""
        # 分句作为原子声明的近似
        import re
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _verify_claim_against_sources(self, 
                                       claim: str, 
                                       sources: List[str]) -> str:
        """
        使用NLI模型验证声明是否为源文档支持
        返回: "supported" | "contradicted" | "unverifiable"
        """
        # 实际使用NLI模型（如RoBERTa-MNLI或DeBERTa-NLI）
        # 这里展示调用逻辑
        for source in sources:
            # 简化的验证：检查文本重叠
            if self._has_textual_support(claim, source):
                return "supported"
            # NLI推理
            # result = self.nli_model.predict(premise=source, hypothesis=claim)
            # if result.label == "entailment":
            #     return "supported"
            # elif result.label == "contradiction":
            #     return "contradicted"
        
        return "unverifiable"
    
    def _has_textual_support(self, claim: str, source: str) -> bool:
        """简化的文本支持检查"""
        words = set(claim.lower().split())
        source_words = set(source.lower().split())
        if not words:
            return False
        overlap = len(words & source_words) / len(words)
        return overlap > 0.5
    
    def compute_composite_score(self, 
                                 faithfulness: Dict,
                                 relevance: float,
                                 completeness: Dict) -> Dict:
        """计算综合质量得分"""
        weights = {
            "faithfulness": 0.4,
            "relevance": 0.3,
            "completeness": 0.3
        }
        
        composite = (
            faithfulness["faithfulness_score"] * weights["faithfulness"] +
            relevance * weights["relevance"] +
            completeness["completeness_score"] * weights["completeness"]
        )
        
        return {
            "composite_score": round(composite, 3),
            "grade": self._score_to_grade(composite),
            "breakdown": {
                "faithfulness": faithfulness,
                "relevance": relevance,
                "completeness": completeness
            }
        }
    
    def _score_to_grade(self, score: float) -> str:
        """将分数转化为等级"""
        if score >= 0.90:
            return "A - 优秀"
        elif score >= 0.80:
            return "B - 良好"
        elif score >= 0.70:
            return "C - 合格"
        elif score >= 0.60:
            return "D - 需改进"
        else:
            return "F - 不合格"
```

---

### 9.5.5 多轮对话提示缓存

```python
class PromptCache:
    """
    多轮对话提示缓存
    
    在多轮对话中，系统提示和部分上下文在多轮之间不会变化，
    通过缓存这些静态部分可以显著降低Token消耗和延迟。
    
    缓存策略：
    - 静态部分: 系统提示（缓存命中率 ~100%）
    - 半静态部分: 对话历史摘要（需定期更新）
    - 动态部分: 检索上下文、用户查询（每轮变化）
    """
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.cache = {}
        self.stats = {
            "hits": 0,
            "misses": 0,
            "tokens_saved": 0
        }
    
    def cache_key(self, template_name: str, session_id: str, 
                  part: str) -> str:
        """生成缓存键"""
        return f"{template_name}:{session_id}:{part}"
    
    def get_or_set_system_prompt(self, 
                                  template_name: str,
                                  session_id: str,
                                  build_func: Callable) -> str:
        """
        获取或构建系统提示（带缓存）
        
        API兼容方式：对于支持prompt_caching的API（如Claude API），
        可以在请求中标记可缓存的静态块。
        """
        key = self.cache_key(template_name, session_id, "system")
        if key in self.cache:
            self.stats["hits"] += 1
            return self.cache[key]
        
        system_prompt = build_func()
        self.cache[key] = system_prompt
        self.stats["misses"] += 1
        return system_prompt
    
    def build_with_api_cache(self, 
                              system_prompt: str,
                              context: str, 
                              query: str) -> Dict:
        """
        构建支持API级别缓存的请求
        
        对于支持prompt caching的API (如Claude)，
        可以在请求中设置cache_control断点
        """
        return {
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}  # 标记为可缓存
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<context>{context}</context>\n\n{query}",
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                }
            ]
        }
    
    def invalidate_session(self, session_id: str):
        """清除会话缓存"""
        keys_to_delete = [
            k for k in self.cache 
            if f":{session_id}:" in k
        ]
        for k in keys_to_delete:
            del self.cache[k]
    
    def get_stats(self) -> Dict:
        """获取缓存统计"""
        total = self.stats["hits"] + self.stats["misses"]
        return {
            "hit_rate": self.stats["hits"] / max(1, total),
            "total_requests": total,
            "tokens_saved": self.stats["tokens_saved"],
            "cache_size": len(self.cache)
        }
```

---

## 9.6 面试高频问题

以下是本章内容对应的企业面试常见问题及回答要点：

### Q1: 请解释RAG系统中的"Lost in the Middle"现象，以及如何解决？

**回答要点**：
- 现象描述：LLM在处理长上下文时，对开头(Primacy Bias)和结尾(Recency Bias)的内容关注度最高，中间部分信息容易被"忽略"
- 量化数据：研究表明，当文档位于上下文中间位置时，检索准确率可比首尾位置下降20-30%
- 解决方案：(1) 使用"Lost in Middle"排序策略，将重要文档交替分布在首尾；(2) 分层上下文结构；(3) 多次检索-多次生成，缩小单次上下文

### Q2: Token预算应该如何在不同组件之间分配？

**回答要点**：
- 一般原则：系统提示10-15%，检索上下文50-60%，对话历史20-25%，用户查询+预留输出15-20%
- 动态调整：事实查询减少检索上下文比例；综合分析增大检索上下文；多轮对话增大历史占比
- 预留策略：始终预留至少10%的Token给输出，避免输出被截断

### Q3: 在RAG系统中，如何有效防止LLM产生幻觉？

**回答要点**：
- 第一层：在提示中明确知识边界，强制"仅使用提供的文档"
- 第二层：要求每句事实陈述必须标注来源
- 第三层：明确要求"不知即言不知"，不允许推测
- 第四层：使用NLI模型对输出进行后验证
- 技术组合：来源锚定(Source Grounding) + 反幻觉提示 + 引用验证

### Q4: 如何处理检索到的文档之间的信息矛盾？

**回答要点**：
- 在提示中要求模型识别并标注矛盾信息
- 列出所有矛盾观点及其各自来源
- 根据来源权威性（官方文档 > 社区文档）和时间新鲜度进行加权
- 应用场景中可由用户做最终判断

### Q5: 上下文压缩的常见方法有哪些？各自的适用场景是什么？

**回答要点**：
- LLMLingua：Token级别压缩，基于困惑度，适合一般场景
- 摘要式：信息密度最高，适合探索性查询
- 选择性压缩：查询感知最强，适合明确查询
- 递归压缩：适合超长文档(>32K)
- 选择依据主要看保真度要求和计算成本预算

### Q6: 如何评估提示工程的效果？

**回答要点**：
- 忠实度(Faithfulness)：NLI验证答案是否由源文档支持
- 相关性(Relevance)：BERTScore评测答案与问题的语义相关性
- 完整性(Completeness)：人工标注关键信息点覆盖率
- 引用准确率(Citation Accuracy)：引用位置对应的源文档是否包含声称信息
- 综合使用自动化指标与人工评估

---

## 9.7 企业最佳实践总结

### 9.7.1 上下文构建最佳实践

1. **始终使用结构化格式**: XML标签格式（如`<document>`, `<content>`）可以显著提升模型对上下文结构的理解，特别是对于指令遵循能力强的模型。
2. **实施分层排序**: 结合相关性降序和Lost-in-Middle优化，确保重要信息不被遗漏。
3. **设置相关性阈值**: 相关性得分低于阈值(如<0.5)的文档不应进入上下文，宁可少给也不要给错。
4. **保留文档元数据**: 标题、来源、时间戳等元数据帮助模型更好地理解和引用上下文。
5. **动态Token分配**: 根据查询类型动态调整各部分的Token配额。

### 9.7.2 提示工程最佳实践

1. **模板版本化管理**: 使用Prompt Registry进行版本控制，支持回滚与环境隔离。
2. **A/B测试驱动优化**: 新提示上线前必须在测试环境进行A/B对比测试。
3. **强制引用要求**: 所有事实性回答都应有来源引用，这是反幻觉的最重要防线。
4. **分级约束策略**: 对知识边界、事实性、不确定性表达和自我校验四层约束。
5. **域适配模板**: 不同行业（法律、金融、医疗）使用定制化的提示模板。

### 9.7.3 运维监控最佳实践

1. **上下文利用率监控**: 跟踪实际的Token使用率分布，优化Token分配策略。
2. **引用覆盖率监控**: 监控被引用的文档占检索文档的比例，过低则需要优化。
3. **幻觉率告警**: 当NLI验证发现幻觉率超过阈值(如5%)时触发告警。
4. **提示性能仪表板**: 建立包含忠实度、相关性、完整性、效率的可视化监控仪表板。
5. **持续迭代闭环**: 收集用户反馈和自动评估数据，持续优化提示模板。

---

## 9.8 章节总结

本章从概念、原理、算法、实现到企业实践，系统性地阐述了RAG系统中的上下文构建和提示工程两大核心环节。

**核心要点回顾**：

| 环节 | 核心问题 | 关键技术 | 企业实践 |
|------|---------|---------|---------|
| 上下文打包 | 如何高效组织检索结果 | 结构化格式、Token预算管理 | 自适应Token分配 |
| 上下文排序 | 如何在有限窗口中最大化信息利用 | 相关性排序、分层排序、Lost-in-Middle优化 | 查询感知的动态排序 |
| 上下文压缩 | 如何在信息保真和Token消耗间平衡 | LLMLingua、选择性压缩、递归压缩 | 多级压缩策略 |
| 多轮对话 | 如何管理持续的对话上下文 | 滑动窗口、摘要压缩、层次化窗口 | 会话级缓存 |
| 提示模板 | 如何设计引导模型正确利用知识的提示 | 角色-约束-输出三层架构 | 版本管理+AB测试 |
| 来源引用 | 如何实现答案的可追溯性 | 内联引用、结构化引用、引用验证 | 引用覆盖率监控 |
| 反幻觉 | 如何防止模型产生幻觉 | 来源锚定、知识边界、强制引用 | NLI后验证+告警 |
| CoT+RAG | 如何提升复杂推理能力 | 查询分解、逐步推理、结构化输出 | 根据查询类型动态启用 |

在下一章（第十章：生成优化与后处理）中，我们将深入探讨如何对RAG系统生成的答案进行后处理优化，包括事实性校验、内容过滤、格式标准化和答案融合等技术。

---

*本章编写参考了LLMLingua、Lost in the Middle论文、Anthropic Prompt Caching文档、RAGAS评估框架等学术与工业界成果。*
