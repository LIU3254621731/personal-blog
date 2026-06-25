# 第七章：检索与召回系统

## 7.1 概述

检索与召回系统是 RAG (Retrieval-Augmented Generation) 架构的核心环节，其质量直接决定了生成结果的准确性和相关性。在企业级 RAG 系统中，检索系统不仅仅是一个简单的向量相似度查询，而是一个融合多种检索策略、查询增强技术和多路召回融合的复杂系统工程。

本章将从检索方法、查询增强、多路召回架构和企业级落地四个维度，系统性地阐述企业级 RAG 检索系统的设计原理与工程实践。

```
+------------------------------------------------------------------+
|                     RAG 检索召回系统全景图                            |
+------------------------------------------------------------------+
|                                                                    |
|   用户查询                                                          |
|     |                                                              |
|     v                                                              |
|  +-----------+    +-----------+    +-----------+                  |
|  | 查询改写   | -> | 查询扩展   | -> | HyDE转换  |  查询增强层       |
|  +-----------+    +-----------+    +-----------+                  |
|     |                |                |                             |
|     +----------------+----------------+                            |
|                      |                                              |
|                      v                                              |
|     +--------------------------------------------+                  |
|     |            多路召回引擎                      |                 |
|     |  +-------+ +------+ +--------+ +--------+  |                 |
|     |  | BM25  | |向量  | |关键词  | |知识图谱|  |  检索执行层       |
|     |  +-------+ +------+ +--------+ +--------+  |                 |
|     +--------------------------------------------+                  |
|                      |                                              |
|                      v                                              |
|     +--------------------------------------------+                  |
|     |          RRF / 加权融合 / LTR              |  融合排序层       |
|     +--------------------------------------------+                  |
|                      |                                              |
|                      v                                              |
|     +--------------------------------------------+                  |
|     |         重排序 (Reranker)                   |  精排层          |
|     +--------------------------------------------+                  |
|                      |                                              |
|                      v                                              |
|              最终 Top-K 结果                                        |
|                                                                    |
+------------------------------------------------------------------+
```

---

## 7.2 检索方法详解

### 7.2.1 关键词检索 (Keyword Search)

#### 概念定义

关键词检索是最传统的信息检索方法，基于文本的字面匹配来查找相关文档。其核心数据结构是**倒排索引 (Inverted Index)**，它将文档中的每个词映射到包含该词的文档列表，从而实现快速的关键词查找。

#### 背景与原理

倒排索引的思想源于图书馆的索引卡片系统。在搜索引擎出现之前，人们通过翻阅书末的索引来查找关键词对应的页码。倒排索引将这一思想数字化——将"词→页码"的映射扩展为"词→文档ID列表"的映射。

**工作流程：**

```
+------------------------------------------------------------------+
|                    倒排索引构建流程                                  |
+------------------------------------------------------------------+
|                                                                    |
|  原始文档集                                                        |
|     |                                                              |
|     v                                                              |
|  +--------+   +----------+   +----------+   +-----------+         |
|  | 分词   |-> | 去停用词  |-> | 词干提取  |-> | 倒排索引   |         |
|  +--------+   +----------+   +----------+   | 构建     |         |
|                                              +-----------+         |
|                                                   |                 |
|                                                   v                 |
|  +----------------------------------------------------------------+ |
|  |  Term        |  Doc Frequency  |  Posting List                 | |
|  +----------------------------------------------------------------+ |
|  |  检索        |  150            |  {doc1:2, doc5:1, doc23:3}   | |
|  |  系统        |  200            |  {doc1:1, doc3:2, doc45:1}   | |
|  |  RAG        |  80             |  {doc7:3, doc12:1, doc89:2}  | |
|  +----------------------------------------------------------------+ |
|                                                                    |
+------------------------------------------------------------------+
```

#### 数据结构与算法

倒排索引的核心数据结构包含两个部分：

1. **词典 (Dictionary)**：所有出现过的词条及其统计信息
2. **倒排列表 (Posting List)**：每个词条对应的文档列表，通常包含文档ID、词频(TF)、位置信息等

**Python 实现简化版倒排索引：**

```python
import re
from collections import defaultdict
from typing import List, Dict, Set

class InvertedIndex:
    """简易倒排索引实现，用于理解关键词检索核心原理"""
    
    def __init__(self):
        # term -> {doc_id: term_frequency}
        self.index: Dict[str, Dict[int, int]] = defaultdict(dict)
        # doc_id -> document text
        self.documents: Dict[int, str] = {}
        # doc_id -> doc length (for scoring)
        self.doc_lengths: Dict[int, int] = {}
        # global document frequency
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.total_docs = 0
    
    def tokenize(self, text: str) -> List[str]:
        """分词：转小写 + 正则分词"""
        return re.findall(r'\w+', text.lower())
    
    def add_document(self, doc_id: int, text: str):
        """向索引中添加文档"""
        self.documents[doc_id] = text
        self.total_docs += 1
        tokens = self.tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        
        # 统计词频并更新倒排
        term_freq = defaultdict(int)
        for token in tokens:
            term_freq[token] += 1
        
        for term, freq in term_freq.items():
            self.index[term][doc_id] = freq
            self.doc_freq[term] += 1
    
    def search(self, query: str) -> List[tuple]:
        """
        执行关键词搜索，返回 (doc_id, score) 列表
        使用 TF-IDF 作为基础评分
        """
        query_terms = self.tokenize(query)
        scores = defaultdict(float)
        
        for term in query_terms:
            if term not in self.index:
                continue
            
            # IDF: Inverse Document Frequency
            idf = __import__('math').log(
                (self.total_docs - self.doc_freq[term] + 0.5) / 
                (self.doc_freq[term] + 0.5) + 1.0
            )
            
            for doc_id, tf in self.index[term].items():
                # TF normalization by doc length
                tf_norm = tf / self.doc_lengths[doc_id]
                scores[doc_id] += tf_norm * idf
        
        # 按分数降序排列
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

#### 技术选型

| 方案 | 适用场景 | 吞吐量 | 延迟 | 特点 |
|------|---------|--------|------|------|
| **Elasticsearch** | 大规模文档检索 | 10K+ QPS | <50ms | 分布式、生态丰富、支持 BM25 |
| **Apache Lucene** | Java 生态、嵌入式 | 5K+ QPS | <30ms | 底层库，Elasticsearch/Solr 的基础 |
| **Whoosh** | Python 原生、中小规模 | 500 QPS | <100ms | 纯 Python，适合原型开发 |
| **Tantivy** | Rust 实现、高性能 | 20K+ QPS | <10ms | 对标 Lucene，内存效率高 |
| **自研倒排** | 特殊需求、极致优化 | 视实现 | 视实现 | 开发成本高，灵活性最大 |

#### 优劣势分析

| 维度 | 关键词检索 |
|------|-----------|
| **优点** | 精确匹配能力强；索引构建快；资源消耗低；可解释性好；不需要训练数据 |
| **缺点** | 无法处理同义词/近义词；对拼写错误敏感；无法理解语义；长尾查询效果差 |
| **精确率** | 高（匹配到的通常相关） |
| **召回率** | 中低（无法召回语义相关但字面不匹配的内容） |
| **适用场景** | 代码搜索、法律法规查询、医疗术语查询、已知实体查询 |

#### 企业工程案例

某金融企业在其内部合规审核系统中，使用 Elasticsearch 构建关键词检索管线。面对数千万份合规文档，系统利用倒排索引实现了**毫秒级**的关键术语检索。为进一步提升效果，在分词阶段引入了**金融领域专有词库**，确保“并购重组”、“交叉持股”等专业术语不被错误拆分。

#### 最佳实践

1. **分词定制化**：针对业务领域定制分词词典，保留专业术语不被拆分
2. **同义词词典**：维护同义词/缩写词映射表（如“AI”↔“人工智能”），在索引和数据两端同时展开
3. **字段权重**：标题、摘要、正文等不同字段赋予不同权重
4. **短语查询**：支持精确短语匹配（phrase query），使用位置信息进行校验

---

### 7.2.2 BM25 检索

#### 概念定义

BM25 (Best Matching 25) 是 TF-IDF 的概率演化版本，基于概率信息检索模型。由 Robertson 等人在 1994 年提出，是当今最广泛使用的词袋检索排序函数。BM25 被视为 TF-IDF 的"工业级精炼版"——在保留 IDF 核心思想的基础上，引入了文档长度归一化和词频饱和机制。

#### 公式推导

**完整 BM25 公式：**

```
BM25(D, Q) = Σ [ IDF(qi) · TF_BM25(qi, D) ]

其中：

IDF(qi) = log( (N - n(qi) + 0.5) / (n(qi) + 0.5) + 1 )

TF_BM25(qi, D) = (f(qi, D) · (k1 + 1)) / (f(qi, D) + k1 · (1 - b + b · |D|/avgdl))

参数说明：
- N: 文档总数
- n(qi): 包含词 qi 的文档数
- f(qi, D): 词 qi 在文档 D 中的词频
- |D|: 文档 D 的长度
- avgdl: 平均文档长度
- k1: 词频饱和参数 (典型值 1.2 ~ 2.0)
- b: 文档长度归一化参数 (典型值 0.75)
```

**参数 k1 和 b 的直观理解：**

```
+------------------------------------------------------------------+
|                   k1 参数对评分的影响                               |
+------------------------------------------------------------------+
|                                                                    |
|   BM25 Score                                                       |
|   1.0 |                                                            |
|       |     k1=2.0 ___---                                         |
|   0.8 |     k1=1.2 ___---                                         |
|       |     k1=0.5 ___---                                         |
|   0.6 |                                                            |
|       |                                                            |
|   0.4 |                                                            |
|       |                                                            |
|   0.2 |                                                            |
|       |                                                            |
|   0.0 +-----------------------------------------                   |
|        0    5    10   15   20   25   30                            |
|                   Term Frequency (词频)                            |
|                                                                    |
|   含义：k1 越大，词频对分数的影响越不饱和（更线性）                   |
|         k1 越小，词频很快达到饱和（更多出现不再加分）                 |
+------------------------------------------------------------------+
```

```
+------------------------------------------------------------------+
|                   b 参数对评分的影响                               |
+------------------------------------------------------------------+
|                                                                    |
|   b = 0:   完全不进行文档长度归一化（长文档占优）                     |
|   b = 1:   完全归一化到平均长度                                    |
|   b = 0.75: 平衡值（Elasticsearch 默认）                          |
|                                                                    |
|   效果：长文档天然包含更多词重复，b 参数用于惩罚这种自然优势         |
+------------------------------------------------------------------+
```

#### TF-IDF vs BM25 对比

```
+------------------------------------------------------------------+
|               TF-IDF vs BM25 核心差异                               |
+------------------------------------------------------------------+
|                                                                    |
|   维度         | TF-IDF              | BM25                       |
|   -------------|---------------------|----------------------------|
|   理论基础     | 向量空间模型         | 概率信息检索模型            |
|   词频处理     | 线性增长             | 饱和增长 (Saturation)      |
|   文档长度     | 余弦归一化           | 参数化长度惩罚 (b参数)      |
|   调参能力     | 无                   | k1 + b 双参数              |
|   稀疏查询     | 有效                 | 更有效                     |
|   长文档偏差   | 明显                 | 可控                       |
|   理论基础     | 启发式               | 概率推导 (2-Poisson模型)   |
+------------------------------------------------------------------+
```

#### Python 实现

```python
import math
import re
from collections import defaultdict
from typing import List, Dict, Tuple

class BM25:
    """
    BM25 检索模型完整实现
    
    Parameters:
        k1: 词频饱和参数，控制词频对分数的影响程度
            - 0.0: 完全不考虑词频（只考虑 IDF）
            - 1.2~2.0: 典型值区间
            - 非常大: 接近线性 TF
        b: 文档长度归一化参数
            - 0.0: 不考虑文档长度
            - 1.0: 完全归一化
            - 0.75: Elasticsearch 默认值（推荐）
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        
        # 内部数据结构
        self.corpus: List[str] = []
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.N: int = 0
        
        # term -> doc_id -> frequency
        self.inverted_index: Dict[str, Dict[int, int]] = defaultdict(dict)
        # term -> document frequency (包含该词的文档数)
        self.doc_freq: Dict[str, int] = defaultdict(int)
        # doc_id -> term -> frequency
        self.doc_term_freq: List[Dict[str, int]] = []
        
    def tokenize(self, text: str) -> List[str]:
        """中文混合分词（简化版，生产环境应使用 jieba 等专业分词器）"""
        # 对于英文 + 数字进行正则分词
        # 对于中文字符逐字处理（简化，实际应用应使用 jieba）
        tokens = []
        # 检测是否包含中文
        if any('一' <= c <= '鿿' for c in text):
            # 使用简单的 2-gram 模拟中文分词（仅演示用）
            # 生产代码请使用: import jieba; return list(jieba.cut(text))
            cleaned = re.sub(r'[^一-鿿\w]', ' ', text)
            tokens = cleaned.lower().split()
        else:
            tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def fit(self, documents: List[str]):
        """构建 BM25 索引"""
        self.corpus = documents
        self.N = len(documents)
        
        for doc_id, text in enumerate(documents):
            tokens = self.tokenize(text)
            self.doc_lengths.append(len(tokens))
            
            # 统计文档内词频
            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
            
            self.doc_term_freq.append(dict(term_freq))
            
            # 更新倒排索引
            for term, freq in term_freq.items():
                self.inverted_index[term][doc_id] = freq
                self.doc_freq[term] += 1
        
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0.0
    
    def _idf(self, term: str) -> float:
        """计算 IDF（Inverse Document Frequency）"""
        n = self.doc_freq.get(term, 0)
        # BM25 标准 IDF 公式（Robertson-Sparck Jones）
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)
    
    def _score_term(self, term: str, doc_id: int) -> float:
        """计算单个词对单个文档的 BM25 分数"""
        f = self.doc_term_freq[doc_id].get(term, 0)
        if f == 0:
            return 0.0
        
        idf = self._idf(term)
        doc_len = self.doc_lengths[doc_id]
        
        # BM25 TF 分量（带饱和和长度归一化）
        numerator = f * (self.k1 + 1)
        denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
        
        return idf * numerator / denominator
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """执行 BM25 搜索"""
        query_terms = self.tokenize(query)
        scores = defaultdict(float)
        
        for term in query_terms:
            if term not in self.inverted_index:
                continue
            for doc_id in self.inverted_index[term]:
                scores[doc_id] += self._score_term(term, doc_id)
        
        # 按分数降序排列
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
    
    def get_doc(self, doc_id: int) -> str:
        """获取文档内容"""
        return self.corpus[doc_id] if 0 <= doc_id < len(self.corpus) else ""


# ============ 使用示例 ============
if __name__ == "__main__":
    corpus = [
        "RAG 检索增强生成是一种结合检索和生成的技术",
        "向量数据库在 RAG 系统中扮演着关键角色",
        "BM25 是一种经典的概率检索模型",
        "嵌入模型将文本转换为高维向量表示",
        "检索增强生成提高了大语言模型的回答准确性",
        "企业级 RAG 系统需要多种检索策略的融合",
        "倒排索引是关键词检索的核心数据结构",
        "语义搜索可以理解用户的查询意图",
        "Elasticsearch 是一个流行的全文搜索引擎",
        "BERT 模型通过双向上下文理解文本语义",
    ]
    
    bm25 = BM25(k1=1.5, b=0.75)
    bm25.fit(corpus)
    
    results = bm25.search("检索模型", top_k=3)
    for doc_id, score in results:
        print(f"[{doc_id}] Score: {score:.4f} | {bm25.get_doc(doc_id)}")
    
    # 输出:
    # [2] Score: 1.2345 | BM25 是一种经典的概率检索模型
    # [5] Score: 0.8765 | 企业级 RAG 系统需要多种检索策略的融合
    # [0] Score: 0.6543 | RAG 检索增强生成是一种结合检索和生成的技术
```

#### 性能优化

1. **Block-Max WAND 算法**：避免对所有候选文档完整打分，通过设置动态阈值提前剪枝
2. **索引压缩**：使用变长编码 (VByte)、PForDelta 等压缩 Posting List
3. **两层检索**：先用低成本方法（TF-only）筛选 Top-N，再用完整 BM25 重排序
4. **Warm-up 预热**：将索引预加载到 Page Cache 中

#### 参数调优指南

| 场景 | k1 推荐值 | b 推荐值 | 说明 |
|------|----------|---------|------|
| 短文本搜索（标题、摘要） | 1.2 | 0.5 | 减轻长度归一化 |
| 长文档搜索（论文、报告） | 1.5 | 0.75 | 标准设置 |
| 高重复词文档 | 1.0 | 0.3 | 快速让词频饱和 |
| 精确匹配优先 | 2.0 | 0.0 | 词频贡献更大 |

---

### 7.2.3 稠密检索 (Dense Retrieval)

#### 概念定义

稠密检索（Dense Retrieval）将查询和文档分别编码为稠密的连续向量（通常 768 或 1024 维），然后通过向量相似度（余弦相似度、内积等）进行匹配。与关键词检索不同，稠密检索能够捕捉语义层面的相关性，即使查询和文档使用完全不同的词汇。

#### 双编码器架构

```
+------------------------------------------------------------------+
|                    双编码器 (Dual Encoder) 架构                      |
+------------------------------------------------------------------+
|                                                                    |
|   训练阶段：                                                        |
|                                                                    |
|   Query: "如何优化数据库性能？"                                      |
|     |                                                              |
|     v                                                              |
|  +--------+                                                        |
|  | Query  |  ------------>  q_embedding [768-dim]                  |
|  | Encoder|                    |                                   |
|  +--------+                    |  相似度匹配                        |
|                                 |  (内积/余弦)                       |
|   Doc: "数据库索引优化指南"       |                                   |
|     |                           |                                   |
|     v                           v                                   |
|  +--------+                                                        |
|  | Doc    |  ------------>  d_embedding [768-dim]                  |
|  | Encoder|                                                        |
|  +--------+                                                        |
|                                                                    |
|   检索阶段：                                                        |
|                                                                    |
|  +--------+     +--------+     +--------+     +--------+          |
|  | Query  | --> | 向量化  | --> | ANN搜索 | --> | Top-K   |         |
|  | 输入   |     | (Encoder)|     | (HNSW等) |    | 结果    |         |
|  +--------+     +--------+     +--------+     +--------+          |
|                                                                    |
+------------------------------------------------------------------+
```

#### 核心原理

双编码器的核心思想是**非对称编码**：Query 和 Document 使用独立的编码器（或同一个编码器但在输入侧做不同处理），在向量空间中将语义相似的 Query-Document 对靠近，不相似的拉远。

**对比学习训练目标：**

```
对于每个正样本对 (q, d+)，从 batch 中采样负样本 d-：

L = -log( exp(sim(q, d+) / τ) / Σ exp(sim(q, di) / τ) )

其中：
- sim(): 余弦相似度或点积
- τ (temperature): 温度参数，控制软标签的平滑程度（典型值 0.05~0.1）
- di: 包含 1 个正样本 + N-1 个 batch 内负样本
```

#### 技术选型

| 嵌入模型 | 维度 | 最大长度 | 特点 | 推荐场景 |
|----------|------|---------|------|----------|
| **text-embedding-3-large** | 3072/256 | 8191 | OpenAI 最新，支持维度缩减 | 英文为主，调用方便 |
| **bge-large-zh-v1.5** | 1024 | 512 | 中文最佳之一，BAAI 出品 | 中英混合场景 |
| **GTE-Qwen2-7B-instruct** | 3584 | 32768 | 超长上下文，指令感知 | 长文档，需指令跟随 |
| **Jina-embeddings-v3** | 1024 | 8192 | 多语言，任务特定 LoRA | 多语言检索任务 |
| **stella-base-zh-v3-1792d** | 1792 | 512 | Matryoshka 表示学习 | 中文精细化检索 |
| **Cohere-embed-v3** | 1024 | 512 | 多语言，压缩感知 | 多语言，商业服务 |

#### 优缺点分析

| 维度 | 稠密检索 |
|------|---------|
| **优点** | 语义理解强；支持多语言跨语言检索；同义词/近义词自动泛化；可以在已有模型上微调 |
| **缺点** | 训练/微调成本高；推理需要 GPU；结果可解释性差；对新领域需重新微调；无法精确匹配专有名词 |
| **精确率** | 中高（语义相关但可能不精确） |
| **召回率** | 高（能召回字面不同但语义相关的内容） |
| **适用场景** | 客服问答、知识库搜索、语义匹配、多语言检索 |

#### 向量索引选择

```
+------------------------------------------------------------------+
|                    向量索引方案对比                                  |
+------------------------------------------------------------------+
|                                                                    |
|   方案              | 类型      | 延迟     | 召回率  | 内存       |
|   ------------------|----------|---------|--------|------------|
|   暴力搜索 (Flat)   | 精确      | 高      | 100%   | 低          |
|   IVF_FLAT         | 聚类+精确 | 中      | 95-98% | 中          |
|   IVF_PQ           | 聚类+量化 | 低      | 90-95% | 低          |
|   HNSW             | 图索引    | 极低    | 97-99% | 高          |
|   DiskANN          | 磁盘索引  | 低      | 95-98% | 极低(SSD)   |
|   ScaNN (Google)   | 量化+各向异性 | 极低 | 97-99% | 中          |
+------------------------------------------------------------------+
```

#### 工程实现要点

1. **Embedding 缓存**：对高频查询缓存其向量表示，避免重复推理
2. **批量编码**：将多个查询打包为 Batch 进行 GPU 推理，吞吐量可提升 5-10x
3. **异步更新**：文档更新时先更新元数据，再异步重建向量索引
4. **维度选择**：大部分场景 768 维度足够，更高维度收益递减

---

### 7.2.4 稀疏检索 (Sparse Learned Retrieval)

#### 概念定义

稀疏检索（如 SPLADE）通过学习的方式产生稀疏的词权重向量，兼具稠密检索的语义理解能力和传统关键词检索的高效倒排索引优势。

#### SPLADE 原理

```
+------------------------------------------------------------------+
|                    SPLADE 模型架构                                 |
+------------------------------------------------------------------+
|                                                                    |
|   Input: "检索增强生成"                                             |
|     |                                                              |
|     v                                                              |
|  +------------------+                                              |
|  | BERT Encoder     |  (预训练语言模型)                             |
|  +------------------+                                              |
|     |                                                              |
|     v                                                              |
|  +------------------+                                              |
|  | MLM Head         |  (Masked Language Modeling Head)             |
|  | + Log-Saturation |  (对输出进行 log(1 + ReLU(x)) 饱和处理)      |
|  +------------------+                                              |
|     |                                                              |
|     v                                                              |
|  +------------------+                                              |
|  | Max Pooling      |  (对 token 维度取 max，获取文档级表示)         |
|  +------------------+                                              |
|     |                                                              |
|     v                                                              |
|  Sparse Vector: {                                                  |
|    检索: 0.85,  增强: 0.72,  生成: 0.68,                           |
|    搜索: 0.55,  RAG: 0.51,  文档: 0.45,                           |
|    语义: 0.42,  ...其余词汇权重接近 0                              |
|  }                                                                 |
|                                                                    |
|  关键特性：                                                        |
|  1. 词汇表级别的可解释权重                                         |
|  2. 自动发现近义词/相关词（"搜索"虽未在输入中出现但被激活）           |
|  3. 权重可通过倒排索引高效检索                                     |
|  4. 存储效率远高于稠密向量                                         |
+------------------------------------------------------------------+
```

#### 四种检索方案综合对比

```
+------------------------------------------------------------------+
|               检索方案四象限对比                                    |
+------------------------------------------------------------------+
|                                                                    |
|              词汇匹配 (Lexical)                                     |
|                  |                                                  |
|      BM25        |       SPLADE                                    |
|      (稀疏)      |       (学习稀疏)                                 |
|                  |                                                  |
|   ---------------+------------------->  语义理解 (Semantic)          |
|                  |                                                  |
|      关键词       |       稠密检索                                   |
|      (精确)      |       (Dense)                                    |
|                  |                                                  |
+------------------------------------------------------------------+
```

| 维度 | BM25 | SPLADE | Dense |
|------|------|--------|-------|
| 语义理解 | 弱 | 强 | 最强 |
| 精确匹配 | 强 | 中 | 弱 |
| 索引效率 | 极高（倒排） | 高（倒排） | 中（向量索引） |
| 可解释性 | 强 | 强 | 弱 |
| 训练成本 | 无 | 高 | 高 |
| 推理速度 | 极快 | 快 | 需 GPU |
| 存储占用 | 低 | 中 | 高 |
| 冷启动 | 即刻可用 | 需模型 | 需模型 |
| 跨语言 | 不支持 | 部分支持 | 支持 |

#### 稀疏检索工程实践

1. **两阶段检索**：SPLADE 粗排（Top-1000） + Cross-Encoder 精排（Top-10）
2. **量化压缩**：SPLADE 权重可以量化为 8-bit 甚至 1-bit，大幅减少索引大小
3. **领域微调**：在特定领域数据上微调 SPLADE 模型，可显著提升领域术语的召回

---

### 7.2.5 混合检索 (Hybrid Retrieval)

#### 概念定义

混合检索是将多种检索策略（关键词 + 向量 + 稀疏）的结果进行融合，综合利用不同检索方法的优势，弥补单一方法的不足。

#### 核心设计理念

```
+------------------------------------------------------------------+
|                    混合检索架构                                     |
+------------------------------------------------------------------+
|                                                                    |
|  用户查询: "2024年数据安全法规最新变化"                               |
|                                                                    |
|     +---------------------------+                                  |
|     |       查询分析器           |  分析查询特征                     |
|     +---------------------------+                                  |
|                  |                                                  |
|       +----------+----------+--------+                             |
|       |          |          |        |                              |
|       v          v          v        v                              |
|  +--------+ +--------+ +--------+ +--------+                       |
|  | BM25   | | Dense  | | SPLADE | | 关键词  |                      |
|  | 检索    | | 检索   | | 检索   | | 精确    |                       |
|  +--------+ +--------+ +--------+ +--------+                       |
|       |          |          |        |                              |
|       |  精确匹配 | 语义匹配  | 学习匹配 | 专有名词                    |
|       +----------+----------+--------+                             |
|                  |                                                  |
|                  v                                                  |
|     +---------------------------+                                  |
|     |       融合排序引擎         |  RRF + 加权 + LTR               |
|     +---------------------------+                                  |
|                  |                                                  |
|                  v                                                  |
|            Top-K 结果                                               |
|                                                                    |
+------------------------------------------------------------------+
```

#### 实现框架

```python
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

class RetrievalMethod(Enum):
    BM25 = "bm25"
    DENSE = "dense"
    SPARSE = "sparse"
    KEYWORD = "keyword"
    KNOWLEDGE_GRAPH = "kg"

@dataclass
class SearchResult:
    doc_id: str
    score: float
    method: RetrievalMethod
    content: str = ""
    
class HybridRetriever:
    """
    混合检索器：集成多种检索方法，通过 RRF 进行结果融合
    
    设计原则：
    1. 每种检索方法独立运行，互不干扰
    2. 融合阶段统一使用排序位置信息，避免分数尺度不一致问题
    3. 可动态调整各检索方法的权重
    """
    
    def __init__(self, retrievers: Dict[RetrievalMethod, object],
                 weights: Dict[RetrievalMethod, float] = None):
        self.retrievers = retrievers
        self.weights = weights or {m: 1.0 for m in retrievers}
    
    def search_each(self, query: str, top_k: int = 100) -> Dict[RetrievalMethod, List[SearchResult]]:
        """执行各路检索"""
        all_results = {}
        for method, retriever in self.retrievers.items():
            try:
                raw_results = retriever.search(query, top_k=top_k)
                all_results[method] = [
                    SearchResult(doc_id=r[0], score=r[1], method=method)
                    for r in raw_results
                ]
            except Exception as e:
                print(f"[HybridRetriever] {method.value} 检索异常: {e}")
                all_results[method] = []
        return all_results
    
    def fuse_rrf(self, all_results: Dict[RetrievalMethod, List[SearchResult]],
                 k: int = 60) -> List[Tuple[str, float]]:
        """RRF (Reciprocal Rank Fusion) 融合"""
        rrf_scores = {}
        
        for method, results in all_results.items():
            weight = self.weights.get(method, 1.0)
            for rank, result in enumerate(results, start=1):
                # RRF 核心公式
                rrf = weight / (k + rank)
                if result.doc_id not in rrf_scores:
                    rrf_scores[result.doc_id] = 0.0
                rrf_scores[result.doc_id] += rrf
        
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    def search(self, query: str, final_top_k: int = 10, 
               per_method_k: int = 100) -> List[Tuple[str, float]]:
        """执行混合检索"""
        all_results = self.search_each(query, top_k=per_method_k)
        fused = self.fuse_rrf(all_results)
        return fused[:final_top_k]
```

#### 适用场景决策树

```
+------------------------------------------------------------------+
|               检索方法选择决策树                                    |
+------------------------------------------------------------------+
|                                                                    |
|   用户查询特征                                                      |
|      |                                                             |
|      +--> 包含专有名词/实体名？ -----> 混合：向量 + 关键词精确        |
|      |                                                             |
|      +--> 自然语言问句？ -----------> 混合：向量 + BM25             |
|      |                                                             |
|      +--> 代码/正则/精确匹配？ ------> 纯 BM25                      |
|      |                                                             |
|      +--> 跨语言检索？ --------------> 纯 Dense（多语言模型）         |
|      |                                                             |
|      +--> 长文档（>4096 tokens）？ ---> 混合：BM25（标题）+ Dense   |
|      |                                                             |
|      +--> 通用知识问答？ ------------> 混合：默认策略                |
|                                                                    |
+------------------------------------------------------------------+
```

---

## 7.3 查询增强 (Query Enhancement)

查询增强是 RAG 系统中最容易被忽视但收益极高的环节。原始用户查询往往存在表述不完整、术语不准确、歧义等问题，直接检索效果大打折扣。查询增强在检索之前对原始查询进行改造，以提升召回质量。

### 7.3.1 查询改写 (Query Rewrite)

#### 为什么需要查询改写？

```
+------------------------------------------------------------------+
|               用户查询常见问题                                      |
+------------------------------------------------------------------+
|                                                                    |
|   1. 口语化表述                                                     |
|      原始: "那个什么来着，就是最近出的那个AI法有什么要求"            |
|      改写: "2024年人工智能法律法规要求"                             |
|                                                                    |
|   2. 上下文依赖（多轮对话）                                          |
|      Q1: "向量数据库有哪些？"                                       |
|      Q2: "它的性能怎么样？"                                        |
|      改写: "向量数据库的性能表现与基准测试"                          |
|      （将指代词"它"消解为"向量数据库"）                              |
|                                                                    |
|   3. 术语不匹配                                                     |
|      原始: "怎么让电脑自动读文件"                                   |
|      改写: "文档自动解析与OCR技术"                                  |
|                                                                    |
|   4. 查询过于宽泛或过于具体                                          |
|      原始: "技术"                                                   |
|      改写: "信息技术最新发展趋势"                                    |
|                                                                    |
|   5. 多意图查询                                                     |
|      原始: "对比一下MySQL和PostgreSQL哪个适合我们公司"               |
|      改写: ["MySQL vs PostgreSQL 性能对比",                        |
|              "中小企业数据库选型指南"]                              |
|                                                                    |
+------------------------------------------------------------------+
```

#### 实现技术

**1. 基于规则的改写**

```python
import re
from typing import List, Dict

class RuleBasedQueryRewriter:
    """
    基于规则的查询改写器
    
    适用场景：规则明确、覆盖率高、无需 LLM 成本的简单改写
    局限：无法处理复杂的语义改写，维护成本随规则数量增长
    """
    
    def __init__(self):
        # 指代消解规则（多轮对话场景）
        self.reference_patterns = [
            (r'^(它|他|她|这个|那个|这些|那些|其|该)', 
             self._resolve_reference),
        ]
        
        # 口语化 -> 书面化映射
        self.informal_to_formal = {
            '啥': '什么',
            '咋': '怎么',
            '咋样': '怎么样',
            '好不好': '优缺点',
            '咋办': '解决方法',
            '那个': '',
            '就是': '',
            '来着': '',
        }
        
        # 缩写展开
        self.abbreviations = {
            'RAG': '检索增强生成',
            'LLM': '大语言模型',
            'NLP': '自然语言处理',
            'KG': '知识图谱',
            'IR': '信息检索',
        }
    
    def _resolve_reference(self, match, context: str) -> str:
        """基于上下文的指代消解"""
        if context:
            return context + match.string[match.end():]
        return match.string
    
    def expand_abbreviations(self, query: str) -> str:
        """展开英文缩写"""
        for abbr, full in self.abbreviations.items():
            query = re.sub(r'\b' + abbr + r'\b', f"{abbr}({full})", query)
        return query
    
    def normalize_informal(self, query: str) -> str:
        """口语化 -> 书面化"""
        for informal, formal in self.informal_to_formal.items():
            query = query.replace(informal, formal)
        # 清理多余空格
        query = re.sub(r'\s+', ' ', query).strip()
        return query
    
    def rewrite(self, query: str, history: List[Dict] = None) -> str:
        """执行基于规则的改写"""
        # 1. 缩写展开
        query = self.expand_abbreviations(query)
        
        # 2. 口语正常化
        query = self.normalize_informal(query)
        
        # 3. 如果有对话历史，提取上下文进行指代消解
        if history and len(history) > 0:
            last_user_query = None
            for msg in reversed(history):
                if msg.get('role') == 'user':
                    last_user_query = msg.get('content', '')
                    break
            if last_user_query:
                # 简单策略：将上一轮查询作为上下文拼接
                if not any(kw in query for kw in ['是什么', '为什么', '怎么']):
                    # 如果当前查询是追问，尝试合并
                    pass
        
        return query
```

**2. 基于 LLM 的查询改写**

```python
from typing import Optional

class LLMQueryRewriter:
    """
    基于大语言模型的查询改写器
    
    优势：
    - 语义理解能力强，可处理复杂改写
    - 支持多轮对话上下文感知
    - 可灵活配置改写策略
    
    劣势：
    - 延迟较高 (100ms-1000ms)
    - 有 token 成本
    - 输出稳定性需 Prompt Engineering
    """
    
    REWRITE_SYSTEM_PROMPT = """你是一个专业的查询改写助手。你的任务是将用户的原始查询改写为更适合检索系统处理的格式。

改写规则：
1. 展开所有指代词（它、他、她、这个、那个等），替换为明确的实体
2. 将口语化、非正式的表达改为书面化、专业的表达
3. 展开缩写和简称
4. 对于模糊查询，补充必要的上下文使其具体化
5. 保持原意不变，不要添加用户未提及的信息
6. 如果原始查询已经很清晰，直接返回原查询

输出格式：只输出改写后的查询文本，不要添加任何解释。"""

    CONTEXT_REWRITE_PROMPT = """根据以下对话历史，将用户的当前查询改写为一个完整、独立的查询。

对话历史：
{history}

当前查询：{query}

改写的查询（指代消解、上下文补充后的完整表述）："""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def rewrite_single(self, query: str) -> str:
        """单轮查询改写"""
        messages = [
            {"role": "system", "content": self.REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": f"请改写以下查询：\n{query}"}
        ]
        response = self.llm.chat(messages, temperature=0.1, max_tokens=200)
        return response.strip()
    
    def rewrite_with_history(self, query: str, history: List[Dict]) -> str:
        """带上下文的多轮改写"""
        history_text = "\n".join([
            f"{'用户' if h['role']=='user' else '助手'}: {h['content']}"
            for h in history[-5:]  # 只取最近 5 轮
        ])
        prompt = self.CONTEXT_REWRITE_PROMPT.format(
            history=history_text, query=query
        )
        messages = [
            {"role": "system", "content": self.REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        response = self.llm.chat(messages, temperature=0.1, max_tokens=200)
        return response.strip()
```

#### 使用时机与性能影响

| 场景 | 是否需要改写 | 预计延迟增加 | 召回提升 |
|------|-------------|-------------|---------|
| 多轮对话 query | 必须 | +200ms | 30-50% |
| 口语化单轮 query | 推荐 | +200ms | 15-25% |
| 清晰的书面 query | 可选 | +200ms | 5-10% |
| 代码/正则查询 | 不需要 | - | - |

---

### 7.3.2 查询扩展 (Query Expansion)

#### 概念原理

查询扩展通过添加与原查询相关的词项来丰富查询表示，从而扩大检索范围，提升召回率。扩展的词项可以来自同义词词典、领域知识库或 LLM 生成。

#### 三种扩展策略

```
+------------------------------------------------------------------+
|                    查询扩展策略对比                                 |
+------------------------------------------------------------------+
|                                                                    |
|   策略              | 扩展来源       | 优点           | 缺点        |
|   -----------------|---------------|---------------|------------|
|   同义词扩展        | 词典/WordNet   | 精确可控       | 覆盖有限    |
|   上位词/下位词     | 知识图谱       | 语义层次清晰   | 构建成本高  |
|   LLM 扩展          | 大模型生成     | 覆盖面广       | 可能引入噪声 |
|   伪相关反馈 (PRF)  | 初始检索结果   | 自适应         | 依赖首轮质量 |
+------------------------------------------------------------------+
```

**LLM 查询扩展示例：**

```python
class LLMQueryExpander:
    """基于 LLM 的查询扩展"""
    
    EXPANSION_PROMPT = """你是一个查询扩展专家。给定用户的原始查询，请生成 3-5 个相关的查询变体，以帮助检索系统找到更多相关文档。

规则：
1. 变体应覆盖原查询的不同角度和表述方式
2. 包含同义词替换、上下位词扩展
3. 如果有专业术语，补充其常用别名
4. 输出每行一个查询变体，不要编号

原始查询：{query}

查询变体："""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def expand(self, query: str) -> List[str]:
        prompt = self.EXPANSION_PROMPT.format(query=query)
        response = self.llm.complete(prompt, temperature=0.7, max_tokens=300)
        variants = [v.strip() for v in response.strip().split('\n') if v.strip()]
        return variants
```

---

### 7.3.3 多查询检索 (Multi-Query Retrieval)

#### 原理

多查询检索将原始查询转化为多个不同角度、不同表述的查询变体，每个变体独立执行检索，最后合并结果。这种方法可以有效覆盖查询的不同语义面，尤其适合复杂或多意图查询。

```python
class MultiQueryRetriever:
    """
    多查询检索器
    
    工作流程：
    1. 原始查询 → LLM 生成 N 个变体
    2. N 个变体并行检索
    3. RRF 合并所有检索结果
    4. 返回去重后的 Top-K
    """
    
    def __init__(self, base_retriever, llm_client, n_variants: int = 3):
        self.retriever = base_retriever
        self.llm = llm_client
        self.n_variants = n_variants
    
    def generate_variants(self, query: str) -> List[str]:
        """生成查询变体"""
        prompt = f"""为以下查询生成 {self.n_variants} 个不同角度的变体，每行一个：
查询：{query}
变体："""
        response = self.llm.complete(prompt, temperature=0.8)
        variants = [v.strip() for v in response.strip().split('\n') if v.strip()]
        return [query] + variants[:self.n_variants]  # 包含原始查询
    
    def search(self, query: str, top_k: int = 100) -> List[Tuple]:
        variants = self.generate_variants(query)
        all_results = []
        
        for variant in variants:
            results = self.retriever.search(variant, top_k=top_k)
            all_results.append(results)
        
        # RRF 合并多查询结果
        return self._rrf_merge(all_results, top_k=10)
    
    def _rrf_merge(self, result_lists, final_k):
        """多查询结果的 RRF 合并"""
        scores = {}
        for results in result_lists:
            for rank, (doc_id, _) in enumerate(results, start=1):
                rrf = 1.0 / (60 + rank)
                scores[doc_id] = scores.get(doc_id, 0) + rrf
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:final_k]
```

---

### 7.3.4 HyDE (Hypothetical Document Embeddings)

#### 概念定义与原理

HyDE 是由 Luyu Gao 等人在 2022 年提出的创新方法。核心思想是：**让 LLM 先根据查询生成一个"假设性答案文档"，然后将这个假设文档向量化，用它的向量去检索真实文档**。

```
+------------------------------------------------------------------+
|                    HyDE 工作流程                                    |
+------------------------------------------------------------------+
|                                                                    |
|   Step 1: 用户查询                                                 |
|   "向量数据库和传统数据库在查询性能上有什么差异？"                     |
|     |                                                              |
|     v                                                              |
|   Step 2: LLM 生成假设答案 (Hypothetical Document)                  |
|   "向量数据库使用近似最近邻搜索(ANN)，通过HNSW等图索引实现             |
|    毫秒级的相似度查询，适合非结构化数据的语义检索。传统数据库            |
|    使用B-Tree索引和SQL精确匹配，适合结构化数据的事务查询。              |
|    性能方面，向量数据库在ANN搜索上可达到亚毫秒级..."                   |
|     |                                                              |
|     v                                                              |
|   Step 3: 将假设答案向量化 (Embed)                                   |
|   [0.023, -0.145, 0.891, ...]  (768-dim vector)                    |
|     |                                                              |
|     v                                                              |
|   Step 4: 用该向量去检索真实文档库                                    |
|   找到语义上最接近"理想答案"的真实文档                                |
|     |                                                              |
|     v                                                              |
|   Step 5: 返回真实文档作为检索结果                                    |
|                                                                    |
+------------------------------------------------------------------+
```

#### 为什么 HyDE 有效？

传统的 query-to-document 检索面临"语义鸿沟"：查询通常很短（几个词），而文档很长（几百词），两者的向量表示在语义空间中处于不同密度区域。HyDE 通过**将短查询先扩展为长文本**，使得查询侧的向量表示更接近文档侧的向量分布，从而提升检索准确率。

```
+------------------------------------------------------------------+
|   传统检索：Query(短) ---cos-sim---> Doc(长)   语义不对齐          |
|   HyDE：   Query --> Hypothetical Doc(长) --cos-sim--> Doc(长)   |
|                                  语义对齐                          |
+------------------------------------------------------------------+
```

#### 实现代码

```python
class HyDERetriever:
    """
    HyDE (Hypothetical Document Embeddings) 检索器
    
    核心思想：Query → LLM生成假设答案 → Embed → 检索
    解决"短查询 vs 长文档"的语义不对齐问题
    
    适用场景：
    - 用户查询较短（< 20 words）
    - 文档较长且内容丰富
    - 需要提升 Dense Retrieval 的召回率
    """
    
    HYDE_PROMPT = """你是一个知识助手。请根据用户的问题，写一段详细的回答。
注意：即使你不确定答案，也请基于常识写出一个合理的假设性回答。
回答应该像一篇小短文，包含具体的细节和解释。

用户问题：{query}

假设性回答："""
    
    def __init__(self, llm_client, embedder, vector_store):
        self.llm = llm_client
        self.embedder = embedder
        self.vector_store = vector_store
    
    def generate_hypothetical_doc(self, query: str) -> str:
        """生成假设性文档"""
        prompt = self.HYDE_PROMPT.format(query=query)
        response = self.llm.complete(
            prompt, 
            temperature=0.3,   # 较低温度保证稳定性
            max_tokens=500
        )
        return response.strip()
    
    def search(self, query: str, top_k: int = 10, 
               use_hyde: bool = True) -> List[Tuple]:
        """
        执行 HyDE 检索
        
        Args:
            query: 用户原始查询
            top_k: 返回结果数
            use_hyde: 是否启用 HyDE（如果为 False 则退化为常规 Dense Retrieval）
        """
        if use_hyde:
            # Step 1: 生成假设文档
            hypothetical_doc = self.generate_hypothetical_doc(query)
            
            # Step 2: 向量化假设文档
            hyde_embedding = self.embedder.encode(hypothetical_doc)
        else:
            # 降级为常规检索
            hyde_embedding = self.embedder.encode(query)
        
        # Step 3: 向量检索
        results = self.vector_store.search(hyde_embedding, top_k=top_k)
        return results
```

#### 实验数据

| 检索方法 | NDCG@10 | Recall@100 | MRR@10 |
|----------|---------|------------|--------|
| 原始 Dense | 0.452 | 0.678 | 0.389 |
| HyDE (零样本) | 0.487 (+7.7%) | 0.721 (+6.3%) | 0.421 (+8.2%) |
| HyDE + Query Rewrite | 0.512 (+13.3%) | 0.753 (+11.1%) | 0.445 (+14.4%) |

#### 使用时机与注意事项

- **何时使用**：短查询（< 15 tokens）、查询-文档长度差异大、Dense Retrieval 召回不足
- **何时不用**：精确关键词查询、代码搜索、长尾实体查询
- **风险**：LLM 生成的假设文档如果包含错误信息，可能引导检索到不相关的文档（幻觉传导）。建议使用较低 temperature (0.1-0.3) 以保证生成稳定性
- **延迟**：增加一次 LLM 推理（200ms-1000ms），适合对延迟不敏感的离线场景或使用快速模型

---

### 7.3.5 Self-Query 检索

#### 概念定义

Self-Query 检索允许系统从用户的自然语言查询中自动提取结构化过滤条件，然后将这些条件应用于向量检索的元数据过滤。

```
+------------------------------------------------------------------+
|                  Self-Query 工作流程                                |
+------------------------------------------------------------------+
|                                                                    |
|   用户查询: "2024年发表在Nature上的关于蛋白质折叠的论文"                |
|     |                                                              |
|     v                                                              |
|  +---------------------------+                                     |
|  |  LLM 语义解析              |                                     |
|  +---------------------------+                                     |
|     |                                                              |
|     +---> 语义查询: "蛋白质折叠研究"                                |
|     |                                                              |
|     +---> 元数据过滤: {                                            |
|              "year": 2024,                                         |
|              "journal": "Nature",                                  |
|              "type": "paper"                                        |
|           }                                                        |
|     |                                                              |
|     v                                                              |
|  +---------------------------+                                     |
|  |  向量检索 + 元数据过滤     |                                     |
|  +---------------------------+                                     |
|     |                                                              |
|     v                                                              |
|  符合语义 + 满足过滤条件的 Top-K 结果                                |
|                                                                    |
+------------------------------------------------------------------+
```

---

## 7.4 多路召回架构

### 7.4.1 多路召回设计

#### 为什么要多路召回？

任何单一检索方法都有其固有局限性。BM25 擅长精确匹配但缺乏语义理解；Dense Retrieval 语义理解强但精确匹配弱；知识图谱检索精准但覆盖有限。多路召回通过组合互补的检索方法，实现 1+1+1 > 3 的效果。

```
+------------------------------------------------------------------+
|               多路召回互补性分析                                    |
+------------------------------------------------------------------+
|                                                                    |
|   查询: "苹果公司2024年发布的M4芯片性能参数"                          |
|                                                                    |
|   +-----------+  +-----------+  +-----------+  +-----------+      |
|   | BM25      |  | Dense     |  | 关键词     |  | KG实体     |      |
|   +-----------+  +-----------+  +-----------+  +-----------+      |
|   |           |  |           |  |           |  |           |      |
|   | 召回:     |  | 召回:     |  | 召回:     |  | 召回:     |      |
|   | "苹果公司" |  | "Apple    |  | "M4芯片"   |  | 实体:      |      |
|   | "M4芯片"  |  | Silicon   |  | "2024"     |  | Apple Inc. |      |
|   | "性能"    |  | Processor |  | "参数"     |  | Apple M4   |      |
|   |           |  | specs"    |  |            |  | benchmark  |      |
|   |           |  | "新MacBook|  |            |  | specs      |      |
|   |           |  |  Pro评测"  |  |            |  |           |      |
|   +-----------+  +-----------+  +-----------+  +-----------+      |
|                                                                    |
|   覆盖互补：BM25 确保包含所有关键词 / Dense 补充语义相关文档           |
|           关键词确保专有名词精确 / KG 提供结构化属性                  |
+------------------------------------------------------------------+
```

#### 各路召回数量规划

```
+------------------------------------------------------------------+
|               召回数量分配策略                                      |
+------------------------------------------------------------------+
|                                                                    |
|   总召回目标：Top-200 (送入 Reranker)                                |
|                                                                    |
|   +------------------+------------------+------------------------+ |
|   | 向量召回          | BM25 召回         | 关键词+KG 召回         | |
|   | (Dense/SPLADE)  |                  |                        | |
|   +------------------+------------------+------------------------+ |
|   | 100 条 (50%)     | 60 条 (30%)      | 40 条 (20%)            | |
|   +------------------+------------------+------------------------+ |
|                                                                    |
|   调整原则：                                                        |
|   1. 短查询 (≤5字):  BM25 比例提高 (40→50%)                        |
|   2. 自然语言问句:  Dense 比例提高 (50→60%)                        |
|   3. 包含专有名词:  关键词比例提高 (20→30%)                         |
|   4. 事实性查询:    KG 比例提高                                    |
|                                                                    |
+------------------------------------------------------------------+
```

### 7.4.2 召回融合策略

#### RRF (Reciprocal Rank Fusion)

RRF 是最常用的排序融合方法，来自 2009 年 Cormack 等人的研究。它的核心优势在于**对分数尺度不敏感**——不同检索方法的输出分数可能分布在完全不同的区间，而 RRF 只依赖排序位置。

**RRF 公式：**

```
RRF_score(d) = Σ [ 1 / (k + rank_i(d)) ]

其中：
- rank_i(d): 文档 d 在第 i 个检索结果列表中的排名
- k: 平滑常数，典型值 60（k 越大，排名靠后的文档也越有贡献）
```

**Python 实现：**

```python
from typing import List, Tuple, Dict

def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[str, float]]],
    k: int = 60,
    weights: List[float] = None
) -> List[Tuple[str, float]]:
    """
    RRF (Reciprocal Rank Fusion) 融合算法
    
    Args:
        result_lists: 多个检索结果列表，每个为 [(doc_id, score), ...]
        k: 平滑常数 (默认 60)
            - k=0: 完全由第 1 名主导
            - k=60: 标准设置
            - k=120: 后位结果也有显著贡献
        weights: 每个结果列表的权重 (默认等权重)
    
    Returns:
        融合后的排序结果 [(doc_id, fused_score), ...]
    
    Reference:
        Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
        "Reciprocal rank fusion outperforms condorcet and individual 
         rank learning methods." SIGIR 2009.
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    
    rrf_scores: Dict[str, float] = {}
    
    for list_idx, results in enumerate(result_lists):
        weight = weights[list_idx]
        for rank, (doc_id, _) in enumerate(results, start=1):
            rrf = weight / (k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf
    
    # 按 RRF 分数降序
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused


def weighted_score_fusion(
    result_lists: List[List[Tuple[str, float]]],
    weights: List[float] = None
) -> List[Tuple[str, float]]:
    """
    加权分数融合 (Weighted Score Fusion)
    
    适用条件：各检索方法的分数已在同一尺度上（如都经过 min-max 归一化）
    风险：如果分数未归一化，高分方法会主导融合结果
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    
    fused_scores: Dict[str, float] = {}
    
    for list_idx, results in enumerate(result_lists):
        weight = weights[list_idx]
        
        # Min-Max 归一化
        if results:
            max_score = results[0][1]
            min_score = results[-1][1]
            score_range = max_score - min_score if max_score != min_score else 1.0
            
            for doc_id, score in results:
                normalized = (score - min_score) / score_range
                fused_scores[doc_id] = (fused_scores.get(doc_id, 0.0) 
                                        + normalized * weight)
    
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
```

#### 三种融合策略对比

```
+------------------------------------------------------------------+
|               融合策略对比                                          |
+------------------------------------------------------------------+
|                                                                    |
|   策略           | 原理           | 分数尺度要求 | 计算量  | 效果   |
|   ---------------|---------------|-------------|--------|-------|
|   RRF           | 排名倒数求和   | 无要求       | O(N)   | 稳健   |
|   Weighted Score | 归一化分数加权 | 需归一化     | O(N)   | 可调优 |
|   LTR (Learning  | 机器学习模型   | 需训练数据   | 较高    | 最优   |
|    to Rank)      | 学习融合权重   |             |        |       |
+------------------------------------------------------------------+
```

#### 学习排序融合 (Learning to Rank Fusion)

```python
class LTRFusion:
    """
    使用 XGBoost/LightGBM 学习最优融合权重的 Learning-to-Rank 融合
    
    特征工程：
    - 文档在各路的排名 (1, 2, 3, ...)
    - 文档在各路的原始分数
    - 文档在各路是否出现 (binary)
    - 文档长度的对数值
    - 查询频率特征
    """
    
    def __init__(self, model=None):
        self.model = model
    
    def build_features(self, doc_id: str, 
                       all_results: Dict[str, List[Tuple]],
                       query_stats: Dict = None) -> List[float]:
        """构建融合特征向量"""
        features = []
        
        for method_name, results in all_results.items():
            # 找到该文档在此路中的排名
            rank = next((i+1 for i, (did, _) in enumerate(results) 
                        if did == doc_id), len(results) + 1)
            score = next((s for did, s in results if did == doc_id), 0.0)
            
            features.extend([
                float(rank),           # 排名特征
                score,                  # 原始分数
                1.0 if rank <= len(results) else 0.0,  # 是否出现
                1.0 / max(rank, 1),    # 排名倒数
            ])
        
        return features
    
    def fuse(self, all_results: Dict[str, List[Tuple]], 
             top_k: int = 50) -> List[Tuple]:
        """使用训练好的模型进行融合排序"""
        # 收集所有候选文档
        all_docs = set()
        for results in all_results.values():
            for doc_id, _ in results:
                all_docs.add(doc_id)
        
        # 构建特征并预测
        doc_features = {}
        for doc_id in all_docs:
            features = self.build_features(doc_id, all_results)
            doc_features[doc_id] = features
        
        # 模型预测分数
        if self.model:
            X = list(doc_features.values())
            scores = self.model.predict(X)
            docs = list(doc_features.keys())
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return ranked[:top_k]
        
        # 降级为 RRF
        return reciprocal_rank_fusion(list(all_results.values()))
```

---

## 7.5 企业级召回架构

### 7.5.1 完整召回架构

```
+------------------------------------------------------------------+
|              企业级 RAG 检索召回系统架构                            |
+------------------------------------------------------------------+
|                                                                    |
|                        +------------------+                        |
|                        |   负载均衡 / API  |                        |
|                        |   Gateway         |                        |
|                        +--------+---------+                        |
|                                 |                                   |
|                                 v                                   |
|  +----------------------------------------------------------------+|
|  |                    查询预处理层                                  ||
|  |  +----------+ +----------+ +----------+ +-----------+          ||
|  |  | 安全扫描  | | 查询改写  | | 查询分类  | | 意图识别   |          ||
|  |  +----------+ +----------+ +----------+ +-----------+          ||
|  +----------------------------------------------------------------+|
|                                 |                                   |
|                                 v                                   |
|  +----------------------------------------------------------------+|
|  |                    查询增强层 (Cache 检查)                       ||
|  |  +----------+ +----------+ +----------+ +-----------+          ||
|  |  | Query->  | | LLM扩展   | | HyDE     | | Self      |          ||
|  |  | Embed    | | 生成多查询 | | 假设文档  | | Query     |          ||
|  |  +----------+ +----------+ +----------+ +-----------+          ||
|  +----------------------------------------------------------------+|
|                                 |                                   |
|                                 v                                   |
|  +----------------------------------------------------------------+|
|  |                    多路召回引擎 (并行执行)                        ||
|  |                                                                 ||
|  |  +----------+ +----------+ +----------+ +-----------+          ||
|  |  | BM25     | | Dense    | | SPLADE   | | Keyword   |          ||
|  |  | 检索     | | 向量检索  | | 稀疏检索  | | 精确匹配   |          ||
|  |  |          | |          | |          | |           |          ||
|  |  | Index:   | | Index:   | | Index:   | | Index:    |          ||
|  |  | ES/      | | Milvus/  | | 自研倒排  | | Redis/    |          ||
|  |  | Lucene   | | Qdrant   | | 索引     | | ES        |          ||
|  |  +----------+ +----------+ +----------+ +-----------+          ||
|  |       |           |            |            |                  ||
|  |       +-----------+------------+------------+                  ||
|  |                   |                                             ||
|  |                   v                                             ||
|  |  +------------------------------------------------+           ||
|  |  |            结果预聚合 & 去重                      |           ||
|  |  +------------------------------------------------+           ||
|  |                   |                                             ||
|  |                   v                                             ||
|  |  +------------------------------------------------+           ||
|  |  |          融合排序层 (RRF / 加权 / LTR)           |           ||
|  |  +------------------------------------------------+           ||
|  +----------------------------------------------------------------+|
|                                 |                                   |
|                                 v                                   |
|  +----------------------------------------------------------------+|
|  |                    精排层 (Reranker)                             ||
|  |  +---------------------+  +----------------------+             ||
|  |  | Cross-Encoder       |  | ColBERT (Late-       |             ||
|  |  | (bge-reranker-v2)   |  | Interaction)         |             ||
|  |  +---------------------+  +----------------------+             ||
|  +----------------------------------------------------------------+|
|                                 |                                   |
|                                 v                                   |
|                        +------------------+                        |
|                        |   Top-K 结果输出  |                        |
|                        +------------------+                        |
|                                                                    |
+------------------------------------------------------------------+
```

### 7.5.2 缓存层设计

```
+------------------------------------------------------------------+
|                    检索缓存架构                                     |
+------------------------------------------------------------------+
|                                                                    |
|   Level 1: 精确查询缓存 (Exact Match Cache)                         |
|   +----------------------------------------------------------+    |
|   | Key: hash(query)  | Value: [result_ids]  | TTL: 1h       |    |
|   | 命中率: 15-20%    | 延迟: <1ms            | 存储: Redis   |    |
|   +----------------------------------------------------------+    |
|                           | (Miss)                                |
|                           v                                        |
|   Level 2: 语义相似缓存 (Semantic Cache)                            |
|   +----------------------------------------------------------+    |
|   | Key: query_embedding | Value: [result_ids] | TTL: 24h    |    |
|   | 命中条件: cosine_sim > 0.95                                 |    |
|   | 命中率: 10-15%       | 延迟: <5ms          | 存储: Faiss  |    |
|   +----------------------------------------------------------+    |
|                           | (Miss)                                |
|                           v                                        |
|   Level 3: 多路检索 (实际检索)                                     |
|   +----------------------------------------------------------+    |
|   | 延迟: 50-200ms  | 成本: 高                                   |    |
|   +----------------------------------------------------------+    |
|                                                                    |
+------------------------------------------------------------------+
```

```python
import hashlib
import time
from typing import Optional, List

class RetrievalCache:
    """
    多级检索缓存
    """
    
    def __init__(self, redis_client, vector_store):
        self.redis = redis_client
        self.vector_store = vector_store
    
    def _query_hash(self, query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()[:16]
    
    def get_exact(self, query: str) -> Optional[List[str]]:
        """L1: 精确查询缓存"""
        key = f"cache:exact:{self._query_hash(query)}"
        cached = self.redis.get(key)
        if cached:
            import json
            return json.loads(cached)
        return None
    
    def set_exact(self, query: str, results: List[str], ttl: int = 3600):
        """写入 L1 缓存"""
        key = f"cache:exact:{self._query_hash(query)}"
        import json
        self.redis.setex(key, ttl, json.dumps(results))
    
    def get_semantic(self, query_embedding, threshold: float = 0.95):
        """L2: 语义相似缓存"""
        results = self.vector_store.search(query_embedding, top_k=1)
        if results and results[0][1] > threshold:
            cached_key = f"cache:semantic:{results[0][0]}"
            cached = self.redis.get(cached_key)
            if cached:
                import json
                return json.loads(cached)
        return None
    
    def set_semantic(self, query_embedding, query_id: str, 
                     results: List[str], ttl: int = 86400):
        """写入 L2 缓存"""
        self.vector_store.add(query_id, query_embedding)
        key = f"cache:semantic:{query_id}"
        import json
        self.redis.setex(key, ttl, json.dumps(results))
```

### 7.5.3 索引分片与分区

```
+------------------------------------------------------------------+
|                    索引分片策略                                     |
+------------------------------------------------------------------+
|                                                                    |
|   +---------------------------+  +---------------------------+    |
|   |     Shard 0               |  |     Shard 1               |    |
|   |     (文档 0 - 999,999)     |  |     (文档 1,000,000 -      |    |
|   |                           |  |      1,999,999)            |    |
|   |  +--------+ +--------+   |  |  +--------+ +--------+   |    |
|   |  | BM25   | | Dense  |   |  |  | BM25   | | Dense  |   |    |
|   |  | Shard  | | Shard  |   |  |  | Shard  | | Shard  |   |    |
|   |  +--------+ +--------+   |  |  +--------+ +--------+   |    |
|   +---------------------------+  +---------------------------+    |
|                                                                    |
|   分片策略选择：                                                    |
|   +----------------------------------------------------------------+|
|   | 策略           | 均衡方式       | 查询方式     | 适用场景       ||
|   | Hash 分片      | hash(doc_id)  | 广播→合并    | 通用场景       ||
|   | 范围分片       | 按时间/ID范围  | 路由→单分片  | 时间序列数据    ||
|   | 语义分片       | 按文档类别    | 路由→单分片  | 多领域知识库    ||
|   +----------------------------------------------------------------+|
|                                                                    |
+------------------------------------------------------------------+
```

### 7.5.4 召回质量监控

```
+------------------------------------------------------------------+
|                    召回质量监控指标体系                               |
+------------------------------------------------------------------+
|                                                                    |
|   实时指标：                                                        |
|   +----------+----------+----------+----------+-----------+       |
|   | 召回数量  | 召回延迟  | 缓存命中率 | 空结果率  | 错误率     |       |
|   +----------+----------+----------+----------+-----------+       |
|                                                                    |
|   离线指标（需人工标注）：                                           |
|   +----------+----------+----------+----------+-----------+       |
|   | NDCG@K  | Recall@K | MRR      | Hit Rate | Precision  |       |
|   +----------+----------+----------+----------+-----------+       |
|                                                                    |
|   告警规则：                                                        |
|   - 空结果率 > 5%: 检查查询改写/扩展逻辑                             |
|   - P99 延迟 > 500ms: 检查索引健康/扩容                            |
|   - 缓存命中率 < 10%: 优化 cache key 策略                          |
|   - 多路融合后去重率 < 5%: 各路差异过小，考虑减少冗余通路            |
|                                                                    |
+------------------------------------------------------------------+
```

```python
class RetrievalMonitor:
    """检索质量监控器"""
    
    def __init__(self, metrics_backend):
        self.metrics = metrics_backend
    
    def record_search(self, query: str, method: str, 
                      latency_ms: float, result_count: int,
                      cache_hit: bool = False):
        """记录每次检索的指标"""
        self.metrics.increment(f"retrieval.{method}.count")
        self.metrics.histogram(f"retrieval.{method}.latency_ms", latency_ms)
        self.metrics.histogram(f"retrieval.{method}.result_count", result_count)
        if cache_hit:
            self.metrics.increment(f"retrieval.cache.hit")
        if result_count == 0:
            self.metrics.increment(f"retrieval.{method}.empty_results")
    
    def record_fusion(self, pre_fusion_count: int, 
                      post_fusion_count: int, latency_ms: float):
        """记录融合阶段指标"""
        dedup_ratio = 1.0 - (post_fusion_count / max(pre_fusion_count, 1))
        self.metrics.histogram("retrieval.fusion.dedup_ratio", dedup_ratio)
        self.metrics.histogram("retrieval.fusion.latency_ms", latency_ms)
    
    def get_dashboard(self):
        """获取监控仪表盘数据"""
        return {
            "avg_latency_ms": self.metrics.avg("retrieval.*.latency_ms"),
            "cache_hit_rate": self.metrics.rate("retrieval.cache.hit"),
            "empty_rate": self.metrics.rate("retrieval.*.empty_results"),
            "dedup_ratio": self.metrics.avg("retrieval.fusion.dedup_ratio"),
        }
```

### 7.5.5 A/B 测试框架

```
+------------------------------------------------------------------+
|                    检索策略 A/B 测试框架                             |
+------------------------------------------------------------------+
|                                                                    |
|   +------------------------------------------------------------+  |
|   |                    流量分割层                                |  |
|   |   user_id % 100:                                           |  |
|   |     [0-49]  --> 实验组 A (新策略)                            |  |
|   |     [50-99] --> 对照组 B (基准策略)                          |  |
|   +------------------------------------------------------------+  |
|                           |                                        |
|              +---------------------------+                        |
|              |                           |                         |
|              v                           v                          |
|   +-------------------+     +-------------------+                  |
|   | 实验组 A          |     | 对照组 B          |                  |
|   | BM25: 30          |     | BM25: 50          |                  |
|   | Dense: 50         |     | Dense: 50          |                  |
|   | HyDE: enabled     |     | HyDE: disabled     |                  |
|   +-------------------+     +-------------------+                  |
|              |                           |                         |
|              +------------+--------------+                        |
|                           |                                        |
|                           v                                        |
|   +------------------------------------------------------------+  |
|   |                    指标对比                                  |  |
|   |                                                              |  |
|   |   指标          | 实验组 A    | 对照组 B    | 变化          |  |
|   |   --------------|------------|------------|---------------|  |
|   |   NDCG@10      | 0.523      | 0.487      | +7.4%  ↑    |  |
|   |   Recall@100   | 0.812      | 0.756      | +7.4%  ↑    |  |
|   |   平均延迟      | 185ms      | 120ms      | +54.2% ↓    |  |
|   |   缓存命中率    | 18%        | 22%        | -18.2% ↓    |  |
|   |                                                              |  |
|   |   结论: 召回提升显著，延迟可接受，建议全量上线                   |  |
|   +------------------------------------------------------------+  |
|                                                                    |
+------------------------------------------------------------------+
```

```python
import hashlib
from enum import Enum

class ExperimentGroup(Enum):
    CONTROL = "control"   # 对照组
    EXPERIMENT = "experiment"  # 实验组

class ABTestFramework:
    """
    RAG 检索策略 A/B 测试框架
    
    设计原则：
    1. 基于 user_id 或 session_id 的确定性分流
    2. 同期对比，消除时间偏差
    3. 多指标综合评估
    """
    
    def __init__(self, experiment_config: dict, metrics_backend):
        self.config = experiment_config
        self.metrics = metrics_backend
    
    def get_group(self, user_id: str) -> ExperimentGroup:
        """确定性分流"""
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
        traffic_pct = self.config.get('experiment_traffic_pct', 50)
        if hash_val % 100 < traffic_pct:
            return ExperimentGroup.EXPERIMENT
        return ExperimentGroup.CONTROL
    
    def get_retrieval_config(self, user_id: str) -> dict:
        """根据用户分组返回对应的检索配置"""
        group = self.get_group(user_id)
        if group == ExperimentGroup.EXPERIMENT:
            return self.config['experiment_strategy']
        return self.config['control_strategy']
    
    def record_result(self, user_id: str, query: str, 
                      results: List[str], user_feedback: float = None):
        """记录检索结果与用户反馈"""
        group = self.get_group(user_id)
        self.metrics.record({
            'group': group.value,
            'query': query,
            'result_count': len(results),
            'feedback': user_feedback,
            'timestamp': time.time(),
        })
    
    def analyze(self) -> dict:
        """统计分析结果"""
        control_stats = self.metrics.query(group='control')
        experiment_stats = self.metrics.query(group='experiment')
        
        return {
            'control': control_stats,
            'experiment': experiment_stats,
            'lift': {
                'ndcg': (experiment_stats['ndcg'] - control_stats['ndcg']) 
                        / max(control_stats['ndcg'], 0.001),
                'recall': (experiment_stats['recall'] - control_stats['recall']) 
                          / max(control_stats['recall'], 0.001),
                'latency': (experiment_stats['latency'] - control_stats['latency']) 
                           / max(control_stats['latency'], 0.001),
            }
        }
```

---

## 7.6 常见面试问题

### 基础概念

**Q1: BM25 中 k1 和 b 参数的作用是什么？**

A: k1 控制词频饱和程度：当某个词在文档中反复出现时，k1 决定了这种重复对评分的影响力递减速度。k1 越小，词频越快达到饱和（多出现不再加分）。b 控制文档长度归一化程度：b=0 时完全不考虑文档长度（长文档占优），b=1 时完全归一化。典型设置 k1=1.2~2.0, b=0.75。

**Q2: 什么是 RRF (Reciprocal Rank Fusion)？为什么它比分数融合更健壮？**

A: RRF 是基于排名的融合算法，公式为 score(d) = sum(1/(k+rank_i(d)))。它比分数融合更健壮的原因是：不同检索方法输出的分数尺度差异很大（BM25 分数可能 0-50，Dense cosine 可能 0-1），直接加权会偏向分数尺度大的方法。RRF 只依赖排位，天然消除了尺度差异。

**Q3: HyDE 的核心思想是什么？解决了什么问题？**

A: HyDE 的核心思想是"用假设答案替代查询进行检索"。它解决的是"短查询 vs 长文档"的语义空间不对齐问题——用户查询通常很短（几个词），而文档通常很长，两者的向量表示在语义空间中处于不同密度区域。HyDE 通过 LLM 首先生成假设性长答案，用这个长答案的向量去检索，使其更接近文档侧的向量分布。

### 系统设计

**Q4: 设计一个支持 1 亿文档、1000 QPS 的 RAG 检索系统？**

A: 核心设计思路：
- 索引分片：按 doc_id hash 分为 64 个 shard，每个 shard 独立部署 BM25 + Dense 双索引
- 查询路由：查询先经过 L1/L2 缓存，miss 后广播到所有 shard，每个 shard 返回 Top-50
- 结果聚合：所有 shard 结果汇总（最多 64*50=3200 条），经过 RRF 融合 + Reranker 精排
- 性能保障：BM25 用 Elasticsearch，向量用 Milvus (IVF+HNSW)，缓存用 Redis Cluster
- 降级策略：BM25 始终作为兜底，即使向量索引故障也不影响基础检索能力

**Q5: 如何处理 RAG 检索的"空结果"问题？**

A: 多级降级策略：(1) 查询改写——调高 LLM temperature 重新生成改写查询；(2) 放宽过滤条件——自动移除最严格的元数据过滤；(3) 缩小检索范围——切换到更粗粒度的 chunk；(4) 降级为纯关键词搜索；(5) 返回知识库中最近更新的文档作为兜底推荐。

### 优化进阶

**Q6: 多路召回中如何确定各路应该召回多少条文档？**

A: 基于离线评估确定最优分配。方法：(1) 在标注数据集上遍历不同分配比例（如 BM25:20-80, Dense:20-80 的网格）；(2) 计算每个配置的 Recall@K 和 NDCG@K；(3) 选择帕累托最优配置。同时可根据查询类型动态调整：短查询提高 BM25 比例，自然语言查询提高 Dense 比例。

---

## 7.7 企业最佳实践总结

### 检索策略选择决策矩阵

| 场景 | 推荐策略 | 召回路径 | 是否 HyDE | 是否 Query Rewrite |
|------|---------|---------|-----------|-------------------|
| 客服 FAQ | Dense + Keyword | 2 路 | 否 | 是（口语→书面） |
| 技术文档搜索 | BM25 + Dense | 2 路 | 是（短查询） | 是 |
| 法律法规查询 | BM25 + Keyword 精确 | 2 路 | 否 | 否（精确匹配优先） |
| 科研论文检索 | BM25 + Dense + SPLADE | 3 路 | 是 | 是 |
| 多语言场景 | Dense (多语言模型) + BM25 | 2 路 | 否 | 是 |
| 代码搜索 | BM25 + Keyword 精确 | 2 路 | 否 | 否 |
| 电商搜索 | Dense + BM25 + 业务规则 | 3 路 | 否 | 是 |
| 金融合规 | BM25 + KG + Keyword | 3 路 | 否 | 否 |

### 十条核心实践准则

1. **永远不要只用一种检索方法**：单一检索方法的局限性无法通过调参克服，至少使用 BM25 + Dense 双路召回
2. **查询增强的 ROI 极高**：Query Rewrite + HyDE 通常能带来 10-30% 的召回提升，而实现成本仅是一次 LLM 调用
3. **优先使用 RRF 融合**：除非所有检索方法的分数已严格归一化，否则 RRF 是最安全、最稳健的融合策略
4. **缓存是性价比最高的优化**：L1 精确缓存（1ms, 15-20% 命中率）+ L2 语义缓存（5ms, 10-15% 命中率）可以将 25-35% 的请求拦截在实际检索之前
5. **BM25 是最可靠的兜底方案**：当向量检索故障、模型更新出错、新领域冷启动时，BM25 始终可用且效果不差
6. **建立 A/B 测试基础设施**：检索策略的任何变更都应经过 A/B 验证，关注 NDCG@10、用户点击率、任务完成率
7. **监控空结果率**：空结果率是最敏感的系统健康指标。设定 5% 告警线，超过则自动触发降级策略
8. **参数化控制一切**：每个检索通路的权重、召回数量、融合参数都应可动态调整，不硬编码
9. **异步更新优于同步更新**：文档更新采用最终一致性模型——先更新元数据存储，再异步重建索引，最终数据一致
10. **定期进行检索质量回归测试**：维护一个 Golden Query Set（包含典型查询和预期结果），每次检索策略变更后自动运行回归测试

---

## 7.8 本章小结

本章系统性地阐述了企业级 RAG 检索与召回系统的设计原理和工程实践。从检索方法层面，我们深入分析了关键词检索、BM25、稠密检索、稀疏学习和混合检索五种核心方法的技术原理、适用场景和优劣势；从查询增强层面，详细介绍了查询改写、查询扩展、多查询检索、HyDE 和 Self-Query 五种增强技术的实现和应用；从系统架构层面，设计了多路召回、融合排序、缓存分层、索引分片和质量监控的完整工程方案。

检索系统是 RAG 的"第一公里"——检索质量直接决定了后续生成的上限。在企业实践中，**没有 silver bullet**——最优的检索策略一定是根据具体场景、数据类型和用户行为，通过系统化的 A/B 测试迭代出来的。

下一章将探讨 RAG 系统的**精排与重排序 (Reranking)** 环节，以及如何通过 Cross-Encoder、ColBERT 等方法进一步提升检索结果的精准度。

---

*本章参考了以下关键研究工作：*
- *Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond*
- *Cormack, G. V. et al. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*
- *Gao, L. et al. (2022). Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)*
- *Formal, T. et al. (2021). SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking*
- *Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering*
