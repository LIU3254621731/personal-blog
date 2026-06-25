# 第一章：RAG 技术发展史

> **面向读者**：高级工程师、技术架构师、AI 系统设计者
> **定位**：面试准备 + 企业部署指南 + 系统设计手册
> **字数**：约 15000 字

---

## 目录

1. [开篇：信息检索的进化之路](#开篇信息检索的进化之路)
2. [1.1 传统搜索 (Traditional IR Search Engines)](#11-传统搜索-traditional-ir-search-engines)
3. [1.2 全文检索 (Full-Text Search)](#12-全文检索-full-text-search)
4. [1.3 Elasticsearch](#13-elasticsearch)
5. [1.4 向量检索 (Vector Retrieval)](#14-向量检索-vector-retrieval)
6. [1.5 稠密检索 (Dense Retrieval)](#15-稠密检索-dense-retrieval)
7. [1.6 RAG (Retrieval-Augmented Generation)](#16-rag-retrieval-augmented-generation)
8. [1.7 混合检索 (Hybrid Search)](#17-混合检索-hybrid-search)
9. [1.8 GraphRAG](#18-graphrag)
10. [1.9 Agentic RAG](#19-agentic-rag)
11. [1.10 Multi-Agent RAG](#110-multi-agent-rag)
12. [深度专题](#深度专题)
13. [全章总结](#全章总结)

---

## 开篇：信息检索的进化之路

```
+------------------+    +------------------+    +------------------+
|   传统搜索        | -> |   全文检索        | -> |   Elasticsearch   |
|   (1970s-1990s)  |    |   (1980s-2000s)  |    |   (2010-present) |
+------------------+    +------------------+    +------------------+
        |                        |                        |
        v                        v                        v
+------------------+    +------------------+    +------------------+
|   向量检索        | -> |   稠密检索        | -> |   RAG            |
|   (2013-2017)    |    |   (2018-2020)    |    |   (2020-present) |
+------------------+    +------------------+    +------------------+
                                                         |
                                                         v
                                              +------------------+
                                              |   混合检索        |
                                              |   (2022-present)  |
                                              +------------------+
                                                         |
                                                         v
                                              +------------------+
                                              | GraphRAG         |
                                              | (2023-present)   |
                                              +------------------+
                                                         |
                                                         v
                                              +------------------+
                                              | Agentic RAG      |
                                              | Multi-Agent RAG  |
                                              | (2024-present)   |
                                              +------------------+
```

每一次跃迁都解决了上一代的根本性瓶颈。本章将逐一剖析每个技术的核心原理、工程实现与企业最佳实践。

---

## 1.1 传统搜索 (Traditional IR Search Engines)

### 概念定义

传统信息检索 (Information Retrieval, IR) 是指从大规模文档集合中，根据用户查询找到相关文档的技术体系。其数学本质是：给定查询 q 和文档集合 D，计算相关性函数 f(q, d)，返回排序后的文档列表。

### 历史背景

传统搜索引擎的起源可追溯至 1970 年代：

| 年代 | 里程碑 | 核心贡献 |
|------|--------|---------|
| 1971 | SMART 系统 (Salton) | 向量空间模型的雏形 |
| 1979 | 布尔模型检索 | AND/OR/NOT 逻辑组合 |
| 1980s | 概率检索模型 | 数学化的相关度计算 |
| 1990s | AltaVista, Lycos | 互联网级搜索引擎 |
| 1998 | Google (PageRank) | 链接分析 + 全文检索 |

### 解决的问题

传统搜索解决的核心问题是**从海量非结构化文本中快速定位相关信息**。在互联网出现之前，信息以纸质形式存储；互联网时代，信息爆炸式增长，需要自动化检索手段。

### 工作原理与数据流

```
用户查询 q
    |
    v
[词法分析 & 分词]
    |
    v
[查询扩展 & 改写]
    |
    v
[倒排索引查找] ---------> [倒排索引文件]
    |                          ^
    v                          |
[文档评分 & 排序]     [文档预处理 & 建库]
    |
    v
[结果返回]
```

### 核心算法

**1. 布尔模型 (Boolean Model)**

```
score(q, d) = (t1 AND t2 AND ... AND tn) ? 1 : 0
```

最简单的匹配模型，精确但无法排序。

**2. 向量空间模型 (Vector Space Model, VSM)**

将文档和查询映射为 tf-idf 加权向量：

```
sim(q, d) = cos(q_vec, d_vec) = (q_vec · d_vec) / (||q_vec|| × ||d_vec||)
```

**3. 概率检索模型 (BM25)**

```
BM25(q, d) = Σ IDF(t_i) × [f(t_i,d) × (k1+1)] / [f(t_i,d) + k1 × (1-b+b×|d|/avgdl)]

其中：
- f(t_i, d) : 词 t_i 在文档 d 中的词频
- IDF(t_i)  : 逆文档频率
- k1, b     : 调参参数 (典型取值 k1=1.5, b=0.75)
- |d|       : 文档长度
- avgdl     : 平均文档长度
```

### 技术选型

| 方案 | 适用场景 | 优势 | 局限 |
|------|---------|------|------|
| 布尔模型 | 法律/专利检索 | 精确可控 | 无排序，全或无 |
| VSM | 科研检索 | 语义化，可排序 | 维度灾难 |
| 概率模型/BM25 | 通用搜索 | 数学优美，效果好 | 不考虑词序 |

### 工程案例：Apache Lucene 架构

```java
// 基于 Lucene 的传统搜索引擎核心流程
public class TraditionalSearchEngine {
    private Directory indexDir;
    private IndexWriter writer;
    private IndexSearcher searcher;

    // 建库阶段
    public void indexDocument(String docId, String content) {
        Document doc = new Document();
        doc.add(new StringField("id", docId, Field.Store.YES));
        doc.add(new TextField("content", content, Field.Store.YES));
        writer.addDocument(doc);
    }

    // 检索阶段 - BM25 评分
    public List<SearchResult> search(String queryStr, int topK) {
        QueryParser parser = new QueryParser("content", new StandardAnalyzer());
        Query query = parser.parse(queryStr);
        TopDocs results = searcher.search(query, topK);

        List<SearchResult> output = new ArrayList<>();
        for (ScoreDoc sd : results.scoreDocs) {
            Document doc = searcher.doc(sd.doc);
            output.add(new SearchResult(doc.get("id"), sd.score));
        }
        return output;
    }
}
```

### 性能优化

| 优化策略 | 技术手段 | 效果 |
|---------|---------|------|
| 索引压缩 | 前缀编码、差值编码 | 减少 70% 存储 |
| 查询缓存 | LRU 缓存热点查询 | 延迟降低 40% |
| 分段检索 | 先查标题再查全文 | 延迟降低 60% |
| 跳表加速 | Skip List on Posting List | 合并复杂度 O(m+n) |

### 面试高频问题

**Q1: BM25 与 TF-IDF 的本质区别是什么？**
> A: TF-IDF 是启发式公式，BM25 基于概率检索框架且引入了文档长度归一化因子 (k1, b)，对超长文档有天然惩罚，避免了 TF-IDF 中长文档的主导效应。

**Q2: 倒排索引的构建复杂度是多少？**
> A: 时间复杂度 O(N log N)，N 为词项总数。实际工程中采用内存排序 + 多路归并策略。

### 企业最佳实践

1. **索引分层**：热数据内存索引，冷数据磁盘索引
2. **查询分类**：短查询走精确匹配，长查询走概率模型
3. **A/B 测试**：用 CTR 评估召回效果
4. **监控指标**：P@10, MRR, NDCG@10, 索引吞吐量

---

## 1.2 全文检索 (Full-Text Search)

### 概念定义

全文检索是在非结构化文本中，通过建立倒排索引 (Inverted Index) 实现关键词匹配和排序的技术。它通过 TF-IDF 等加权机制评估查询与文档的相关性。

### 倒排索引原理

倒排索引是全文检索的基石，其核心数据结构：

```
正排索引 (Forward Index)：
doc1 -> [word1, word2, word3, ...]
doc2 -> [word2, word4, word5, ...]

倒排索引 (Inverted Index)：
word1 -> [doc1:tf=2, doc5:tf=1, doc9:tf=3]
word2 -> [doc1:tf=1, doc2:tf=4, doc7:tf=2]
word3 -> [doc1:tf=3, doc3:tf=1]
```

倒排索引的数据流：

```
[文档集合]
    |
    v
[分词器 Tokenizer] ------> [词项流]
    |
    v
[语言处理]                [停用词过滤]
    |                          |
    v                          v
[词干提取 Stemming]      [小写化]
    |
    v
[倒排表构建]
    |
    v
[排序存储到磁盘]
    |
    v
[倒排索引文件]
    格式: [term][doc_freq] -> [doc_id, term_freq, positions...]
```

### TF-IDF 算法详解

**TF (Term Frequency)**：

```
TF(t, d) = count(t, d) / |d|

其中 count(t,d) 是词 t 在文档 d 中出现次数，|d| 是文档总词数。
```

**IDF (Inverse Document Frequency)**：

```
IDF(t) = log( N / df(t) ) + 1

其中 N 是文档总数，df(t) 是包含词 t 的文档数。
```

**TF-IDF**：

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

### Python 实现示例

```python
import math
from collections import Counter, defaultdict

class FullTextSearchEngine:
    """简易全文检索引擎实现 (教学用)"""

    def __init__(self):
        self.inverted_index = defaultdict(list)  # term -> [(doc_id, tf)]
        self.documents = {}                       # doc_id -> text
        self.N = 0

    def tokenize(self, text: str) -> list:
        """简易分词 + 小写化"""
        import re
        return re.findall(r'\w+', text.lower())

    def add_document(self, doc_id: str, content: str):
        """添加文档并更新倒排索引"""
        tokens = self.tokenize(content)
        tf_counter = Counter(tokens)
        self.documents[doc_id] = content
        self.N += 1

        for term, tf in tf_counter.items():
            normalized_tf = tf / len(tokens)
            self.inverted_index[term].append((doc_id, normalized_tf))

    def compute_idf(self, term: str) -> float:
        """计算 IDF"""
        df = len(self.inverted_index.get(term, []))
        return math.log(self.N / df) + 1 if df > 0 else 0

    def search(self, query: str, top_k: int = 10) -> list:
        """TF-IDF 检索"""
        query_terms = self.tokenize(query)
        scores = defaultdict(float)

        for term in query_terms:
            idf = self.compute_idf(term)
            for doc_id, tf in self.inverted_index.get(term, []):
                scores[doc_id] += tf * idf

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score) for doc_id, score in ranked[:top_k]]
```

### 优缺点分析

| 维度 | 优势 | 劣势 |
|------|------|------|
| 查准率 | 关键词精确匹配，准确率高 | 无法处理同义词 |
| 查全率 | 倒排索引覆盖全面 | 无法检索语义相关但用词不同的内容 |
| 性能 | O(1) 词项查找 | 长查询合并开销大 |
| 存储 | 倒排表压缩后空间效率高 | 原始文本仍需存储 |
| 语义 | — | **完全缺失语义理解** |

### 核心痛点

```
查询："如何提升模型训练速度"
文档："深度学习加速技术总结"

传统全文检索: 评分 0  (无共同词项)
人类理解:     高度相关 (语义一致)
```

**这就是为什么我们需要向量检索和 RAG —— 解决语义鸿沟。**

### 面试高频问题

**Q: 倒排索引中的 Positional Index 是什么？**
> A: 除了记录 doc_id 和 term_freq，还额外存储每个词项在文档中的位置列表，用于支持短语查询 (Phrase Query) 和邻近查询 (Proximity Query)。例如 "machine learning" 必须 machine 和 learning 相邻出现。

---

## 1.3 Elasticsearch

### 概念定义

Elasticsearch (ES) 是基于 Apache Lucene 构建的分布式、RESTful 搜索与分析引擎。它将 Lucene 的倒排索引能力打包为集群化的分布式服务，并提供了高级的文本分析、聚合分析和实时搜索能力。

### 架构设计

```
                          [客户端 / Kibana]
                                |
                     [REST API (JSON over HTTP)]
                                |
                    +-----------+-----------+
                    |                       |
              [Master Node]            [Data Node 1]
              (集群管理)                (分片 P0, R1)
                    |                       |
              [Data Node 2]            [Data Node 3]
              (分片 P1, R0)            (分片 P2, R3)
                    |                       |
                    +-----------+-----------+
                                |
                     [协调节点 (Coordinating)]
                        - 请求路由
                        - 结果合并 & 重排序
```

核心组件说明：

| 组件 | 职责 |
|------|------|
| Index | 逻辑命名空间，类似数据库的 Database |
| Shard | Lucene 实例，数据的物理分片 |
| Replica | 分片的副本，提供高可用 |
| Segment | Lucene 内部的不可变索引段 |
| Translog | 写入操作的 WAL 日志，保证数据不丢 |

### Analyzer 分词器体系

```
Input Text: "The quick brown foxes jumped over the lazy dog's bone"
    |
    v
[Character Filter]  --> 去除 HTML 标签、替换特殊字符
    |
    v
[Tokenizer]         --> 按空格/标点切分为 tokens
    |                   ["The","quick","brown","foxes","jumped","over","the","lazy","dog's","bone"]
    v
[Token Filters]     -->
    |   Lowercase: ["the","quick","brown","foxes","jumped","over","the","lazy","dog's","bone"]
    |   Stopwords:  ["quick","brown","foxes","jumped","lazy","dog's","bone"]
    |   Stemming:   ["quick","brown","fox","jump","lazi","dog","bone"]
    v
[Final Tokens]
```

#### 常见 Analyzer 对比

| Analyzer | 分词策略 | 适用语言 | 特点 |
|----------|---------|---------|------|
| standard | Unicode 标准分词 | 通用 | 按词边界切分 |
| ik_max_word | 最细粒度 | 中文 | 穷举所有可能的分词组合 |
| ik_smart | 最粗粒度 | 中文 | 最可能的单一分词结果 |
| kuromoji | 形态素解析 | 日文 | 日本语特有 |
| ngram | 2-gram/3-gram | 无空格语言 | 滑动窗口切分 |

### 相关性评分

ES 默认使用 Lucene 的 **实用评分函数 (Practical Scoring Function)**：

```
score(q, d) = coord(q,d) × queryNorm(q) × Σ [tf(t,d) × idf(t)² × t.getBoost() × norm(t,d)]

其中：
- coord(q,d):   协调因子，匹配词项越多分数越高
- queryNorm(q): 查询归一化因子
- norm(t,d):    字段长度归一化
```

**BM25 评分 (ES 5.0+ 默认)**：

```
BM25(q, d) = Σ IDF(t_i) × [f(t_i,d) × (k1+1)] / [f(t_i,d) + k1 × (1-b+b×|d|/avgdl)]
```

### 部署方案

#### 单机部署 (开发环境)

```yaml
# docker-compose.yml
version: '3.8'
services:
  elasticsearch:
    image: elasticsearch:8.12.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms2g -Xmx2g"
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - es_data:/usr/share/elasticsearch/data

  kibana:
    image: kibana:8.12.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200

volumes:
  es_data:
```

#### 生产集群部署 (3 节点)

```yaml
# 生产环境配置要点
cluster.name: enterprise-search
node.name: ${HOSTNAME}
node.master: true
node.data: true
path.data: /data/elasticsearch
path.logs: /var/log/elasticsearch
network.host: 0.0.0.0
discovery.seed_hosts: ["es01.internal", "es02.internal", "es03.internal"]
cluster.initial_master_nodes: ["es01", "es02", "es03"]

# JVM 配置
# -Xms16g -Xmx16g  (不超过物理内存的 50%)
# -XX:+UseG1GC

# 索引配置
index.number_of_shards: 5
index.number_of_replicas: 1
index.refresh_interval: 30s
index.translog.durability: async
```

### 性能优化 Checklist

| 优化项 | 配置/策略 | 收益 |
|--------|----------|------|
| 堆内存 | 不超过 32GB，设为物理内存的 50% | 避免指针压缩失效 |
| 禁用 swap | `bootstrap.memory_lock: true` | 避免 GC 导致 swap |
| 控制分片大小 | 每个分片 10-50GB | 平衡查询并行度 |
| 字段映射 | 关闭不需要索引的字段 | 减少索引体积 |
| Doc Values | 排序/聚合字段启用 | 基于列式存储加速 |
| 强制合并 | `_forcemerge?max_num_segments=1` | 减少 segment 碎片 |

### 面试高频问题

**Q: ES 的近实时搜索 (Near Real-Time) 原理是什么？**
> A: 文档写入时先到内存 buffer 和 Translog，每秒 refresh 将 buffer 刷成新的 segment（此时数据可搜索）。Translog 每 30 分钟或满了 flush 到磁盘，保证持久性。refresh 间隔默认为 1 秒，所以数据写入后最多 1 秒即可搜索到。

**Q: 如何优化 ES 写入性能？**
> A: (1) 批量索引，bulk_size 设为 5-15MB；(2) 增加 refresh_interval 至 30s；(3) 设为 async translog；(4) 写入期间设 replica=0 再增加；(5) 使用 auto-generated ID 而非自定义 ID。

---

## 1.4 向量检索 (Vector Retrieval)

### 概念定义

向量检索是将文本、图像、音频等非结构化数据映射到高维向量空间，通过向量之间的相似度计算（如余弦相似度、欧氏距离）进行近似最近邻搜索 (Approximate Nearest Neighbor, ANN) 的技术。

### 核心思想

```
文本 -> [Embedding Model] -> 向量 -> [向量空间]
                                       |
查询 -> [Embedding Model] -> 向量 -> [相似度计算] -> Top-K 结果
```

**数学定义**：

```
给定查询向量 q ∈ R^d，数据集 X = {x1, x2, ..., xn} (xi ∈ R^d)
找到 Top-K 个结果: argmin_k dist(q, xi)

常见距离度量：
- 欧氏距离: dist(a,b) = √(Σ(ai - bi)²)
- 余弦相似度: sim(a,b) = (a·b) / (||a|| × ||b||)
- 内积: sim(a,b) = a·b
```

### Annoy / FAISS / HNSW 算法对比

| 算法 | 类型 | 查询复杂度 | 召回率 | 内存占用 | 索引构建 |
|------|------|-----------|--------|---------|---------|
| Brute Force | 精确 | O(n×d) | 100% | O(n×d) | O(1) |
| Annoy | 树结构 | O(log n) | ~95% | O(n×d) | O(n log n) |
| FAISS-IVF | 量化 | O(√n) | ~90% | 压缩 | O(n log n) |
| FAISS-HNSW | 图结构 | O(log n) | ~98% | 高 | O(n log n) |
| ScaNN | 量化+重排 | O(log n) | ~97% | 压缩 | O(n log n) |

### 算法剖析：HNSW (Hierarchical Navigable Small World)

```
层级结构示意：

Layer 2:  * -------- *          (最顶层，节点少，边多)
           \         /
Layer 1:    * --- * --- * --- *  (中间层)
             \   / \   / \   /
Layer 0:      *-*---*-*---*-*   (底层，所有节点，局部连接)
```

**搜索算法 HNSW**：

```
1. entry_point = 顶层随机入口
2. for layer = top_layer downto 1:
3.     entry_point = 在当前层贪心搜索到最近邻
4.     (贪心: 如果邻居更近就移动，直到无法改进)
5.
6. for layer = 0:
7.     从 entry_point 开始，维护一个大小为 ef 的候选集
8.     在候选集的最近邻中贪心扩展
9.     返回 Top-K
```

### FAISS 实战代码

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class FAISSRetriever:
    """基于 FAISS 的向量检索引擎"""

    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5",
                 index_type: str = "IVF"):
        self.encoder = SentenceTransformer(model_name)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.index = None
        self.index_type = index_type
        self.documents = []  # 存储原始文档

    def build_index(self, documents: list):
        """构建 FAISS 索引"""
        self.documents = documents
        embeddings = self.encoder.encode(
            documents, normalize_embeddings=True, show_progress_bar=True
        ).astype('float32')

        if self.index_type == "IVF":     # 倒排文件索引
            quantizer = faiss.IndexFlatIP(self.dimension)
            nlist = min(int(np.sqrt(len(documents))), 4096)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.train(embeddings)

        elif self.index_type == "HNSW":  # 图索引，高召回场景
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 128

        elif self.index_type == "PQ":    # 乘积量化，内存受限场景
            m = 48  # 子向量数
            self.index = faiss.IndexIVFPQ(
                faiss.IndexFlatL2(self.dimension),
                self.dimension, 32, m, 8  # 8-bit 编码
            )
            self.index.train(embeddings)

        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 10) -> list:
        """检索 Top-K 相似文档"""
        query_vec = self.encoder.encode(
            [query], normalize_embeddings=True
        ).astype('float32')

        distances, indices = self.index.search(query_vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # -1 表示未找到
                results.append({
                    "document": self.documents[idx],
                    "score": float(dist),
                    "index": int(idx)
                })
        return results
```

### 技术选型矩阵

| 场景 | 推荐技术 | 原因 |
|------|---------|------|
| < 100万 文档 | FAISS Flat | 精确检索，延迟可控 |
| 100万 - 1000万 | FAISS IVF + PQ | 平衡速度与内存 |
| > 1000万 | ScaNN / DiskANN | 极致压缩与磁盘扩展 |
| 实时更新 | Milvus / Qdrant | 向量数据库支持 CRUD |
| 多模态 | Weaviate | 原生多模态支持 |
| 中文场景 | bge-large-zh 模型 + Milvus | 中文 SOTA + 成熟生态 |

### 面试高频问题

**Q: 为什么需要 ANN？精确 KNN 的瓶颈在哪？**
> A: 精确 KNN 复杂度为 O(n×d)，当 n 达到百万级时单次查询需要数秒。ANN 通过牺牲少量精度（通常 < 5%）将查询复杂度降至 O(log n)，这是工业应用的必然选择。

**Q: FAISS IVF 的倒排文件如何处理未命中？**
> A: IVF 只搜索距查询最近的 nprobe 个聚类中心。如果真正最近邻所在的聚类未命中，则产生召回损失。增大 nprobe 可提升召回但增加延迟——这是精确度与速度的经典 trade-off。

---

## 1.5 稠密检索 (Dense Retrieval)

### 概念定义

稠密检索 (Dense Retrieval) 是指使用深度神经网络（如 BERT 系列）将查询和文档编码为固定的稠密向量（通常 768 维），在同一个向量空间中进行语义相似度匹配的技术。不同于传统的稀疏向量 (TF-IDF/BM25) 依赖词项重叠，稠密检索通过语义编码解决了"词汇鸿沟"问题。

### DPR (Dense Passage Retrieval) 架构

DPR (Karpukhin et al., 2020) 是稠密检索的标志性工作：

```
                           +------------------+
用户查询 q  ------------->| Question Encoder  |-----+
                           | (BERT-based)      |     |
                           +------------------+      |
                                                     v
                                              [内积相似度]
                                                     ^
                           +------------------+      |
文档 d_i  ---------------->| Passage Encoder  |-----+
                           | (BERT-based)      |
                           +------------------+
```

**训练目标 (In-Batch Negatives)**：

```
L(q, p⁺, p⁻₁, ..., p⁻_n) = -log[ exp(sim(q, p⁺)) /
                                     (exp(sim(q, p⁺)) + Σ exp(sim(q, p⁻_i))) ]

其中 p⁺ 是正例文档，p⁻_i 是负例文档（同 batch 内其他正例）。
```

### DPR 核心实现

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class DualEncoder(nn.Module):
    """DPR 双编码器架构"""

    def __init__(self, model_name: str = "bert-base-uncased",
                 projection_dim: int = 768):
        super().__init__()
        self.question_encoder = BertModel.from_pretrained(model_name)
        self.passage_encoder = BertModel.from_pretrained(model_name)
        # 可选：投影层（用于维度对齐或降维）
        self.projection = nn.Linear(
            self.question_encoder.config.hidden_size, projection_dim
        )

    def encode_question(self, input_ids, attention_mask):
        """编码查询"""
        outputs = self.question_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        # 使用 [CLS] token 的表示
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_embedding)

    def encode_passage(self, input_ids, attention_mask):
        """编码文档"""
        outputs = self.passage_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_embedding)

    def forward(self, q_input_ids, q_mask, p_input_ids, p_mask):
        """前向传播，返回查询和文档向量"""
        q_emb = self.encode_question(q_input_ids, q_mask)
        p_emb = self.encode_passage(p_input_ids, p_mask)
        return q_emb, p_emb


class DPRTrainer:
    """DPR 训练循环"""

    def train_step(self, batch, model, optimizer):
        q_emb, p_emb = model(
            batch['q_ids'], batch['q_mask'],
            batch['p_ids'], batch['p_mask']
        )

        # In-batch negatives: 同 batch 内其他 passage 当负例
        batch_size = q_emb.size(0)
        scores = torch.matmul(q_emb, p_emb.T)  # [B, B]
        # 对角线是正例分数
        labels = torch.arange(batch_size, device=scores.device)
        loss = nn.CrossEntropyLoss()(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
```

### 稀疏检索 vs 稠密检索 对比

```
+---------------------+---------------------------+
| 稀疏检索 (BM25)      | 稠密检索 (DPR)            |
+---------------------+---------------------------+
| 词项精确匹配         | 语义模糊匹配              |
| 可解释性强           | 黑盒模型                  |
| 冷启动好（无需训练）  | 需要大量标注数据训练       |
| 词汇鸿沟问题严重      | 解决同义词/语义匹配        |
| 领域迁移好            | 需领域微调                |
| 索引更新快            | 索引更新需重编码          |
+---------------------+---------------------------+
```

### Embedding 模型选型

| 模型 | 维度 | 评测基准 | 中文支持 | 适用场景 |
|------|------|---------|---------|---------|
| text-embedding-3-large | 3072/256/1024 | MTEB Top | 有限 | 通用英文 |
| bge-large-zh-v1.5 | 1024 | C-MTEB SOTA | 优秀 | 中文 RAG |
| GTE-Qwen2-7B-instruct | 3584 | MTEB Leader | 优秀 | 高精度通用 |
| E5-mistral-7b-instruct | 4096 | MTEB SOTA | 一般 | 高精英文 |
| Cohere-embed-v3 | 1024 | 商业 SOTA | 多语言 | 商业场景 |
| jina-embeddings-v3 | 1024 | 较好 | 多语言 | 多模态 Embedding |

### 面试高频问题

**Q: DPR 的 in-batch negatives 有什么优缺点？**
> A: 优点是不需要显式挖掘负例，训练效率高；缺点是可能存在 false negatives（batch 内其他正例被当作负例），batch size 越大效果越好但也对 GPU 内存提出更高要求。

**Q: 稠密检索如何保证对长文档的召回？**
> A: 通常做法：(1) 将长文档切分成多个 passage，每个 passage 独立编码索引；(2) 检索时取命中最高的 passage 对应的文档；(3) 或使用 ColBERT 等 late-interaction 方法保留 token 级交互信息。

---

## 1.6 RAG (Retrieval-Augmented Generation)

### 概念定义

RAG (Retrieval-Augmented Generation) 是由 Lewis 等人在 2020 年发表于 NeurIPS 的论文《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》中正式提出的技术范式。其核心思想是：

> **在 LLM 生成回答之前，先从外部知识库中检索相关信息，将其作为上下文注入生成过程，从而让模型能够基于外部知识回答问题，而非仅依赖训练时固化的参数化知识。**

### Lewis 2020 论文核心架构

```
                         +------------------+
                         | Query Encoder    |
                         | (BERT-based)     |
                         +--------+---------+
                                  |
                                  v
                    +-----------------------------+
                    | MIPS (Maximum Inner Product |
                    |       Search)               |
                    +-----------------------------+
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
          +------------------+         +------------------+
          | Retrieved Doc 1  |   ...   | Retrieved Doc k  |
          +------------------+         +------------------+
                    |                           |
                    +-------------+-------------+
                                  |
                                  v
                    +----------------------------+
                    | Generator (BART/T5/GPT)    |
                    | p(y|x, z1, ..., zk)       |
                    +----------------------------+
                                  |
                                  v
                            [Generated Output]
```

**数学形式**：

```
RAG-Sequence (每个 token 看同一组文档):
p_RAG-Seq(y|x) ≈ Σ_{z ∈ top-k(p_η(·|x))} p_η(z|x) × Π_i p_θ(y_i|x, z, y_{1:i-1})

RAG-Token (每个 token 可看不同文档):
p_RAG-Tok(y|x) ≈ Π_i Σ_{z ∈ top-k(p_η(·|x))} p_η(z|x) × p_θ(y_i|x, z, y_{1:i-1})
```

### RAG 完整数据流

```
Step 1: 数据摄取 & 索引
[文档] -> [分块 Chunking] -> [Embedding Model] -> [向量数据库]

Step 2: 查询 & 检索
[用户查询] -> [Embedding Model] -> [查询向量] -> [ANN 检索]
    -> [Top-K 相关文档块]

Step 3: 上下文组装 & 生成
[系统提示词]
    + [检索到的上下文1]
    + [检索到的上下文2]
    + ...
    + [用户查询]
    -> [LLM] -> [带来源引用的回答]
```

### Python 完整实现

```python
import hashlib
from typing import List, Optional
from dataclasses import dataclass, field

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

@dataclass
class Document:
    """文档数据结构"""
    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)

@dataclass
class RetrievedDoc:
    """检索结果"""
    document: Document
    score: float

class RAGPipeline:
    """标准 RAG Pipeline 实现"""

    def __init__(self,
                 embedding_model: str = "BAAI/bge-large-zh-v1.5",
                 llm_model: str = "gpt-4",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 top_k: int = 5):
        self.encoder = SentenceTransformer(embedding_model)
        self.llm_client = OpenAI()
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

    # ====== Step 1: 文档分块 (Chunking) ======
    def chunk_text(self, text: str) -> List[str]:
        """滑动窗口分块策略"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += (self.chunk_size - self.chunk_overlap)
        return chunks

    # ====== Step 2: 索引构建 ======
    def build_index(self, documents: List[Document]):
        """构建向量索引"""
        self.documents = []
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc.content)
            for i, chunk in enumerate(chunks):
                chunk_doc = Document(
                    doc_id=f"{doc.doc_id}_chunk_{i}",
                    content=chunk,
                    metadata={"parent_id": doc.doc_id, "chunk_index": i}
                )
                self.documents.append(chunk_doc)
                all_chunks.append(chunk)

        self.embeddings = self.encoder.encode(
            all_chunks, normalize_embeddings=True, show_progress_bar=True
        )
        print(f"Indexed {len(self.documents)} chunks from {len(documents)} documents")

    # ====== Step 3: 检索 ======
    def retrieve(self, query: str) -> List[RetrievedDoc]:
        """向量相似度检索"""
        query_emb = self.encoder.encode(
            [query], normalize_embeddings=True
        )
        scores = np.dot(self.embeddings, query_emb.T).flatten()
        top_indices = np.argsort(scores)[-self.top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append(RetrievedDoc(
                document=self.documents[idx],
                score=float(scores[idx])
            ))
        return results

    # ====== Step 4: 上下文组装 ======
    def build_prompt(self, query: str, retrieved: List[RetrievedDoc]) -> str:
        """组装 Prompt"""
        context_parts = []
        for i, rd in enumerate(retrieved):
            context_parts.append(
                f"[来源 {i+1}] (相关度: {rd.score:.3f})\n{rd.document.content}"
            )
        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""你是一个专业的企业知识助手。请基于以下参考文档回答用户问题。

参考文档:
{context}

用户问题: {query}

回答要求:
1. 答案必须基于参考文档中的信息
2. 如果参考文档不足以回答，请明确说明
3. 引用时请标注 [来源 N]
4. 保持回答专业、准确、简洁

回答:"""
        return prompt

    # ====== Step 5: 生成回答 ======
    def generate(self, query: str) -> dict:
        """完整 RAG 流程：检索 + 生成"""
        retrieved = self.retrieve(query)

        if not retrieved:
            return {
                "answer": "抱歉，未找到相关信息。",
                "sources": [],
                "pipeline": "RAG (Retrieval-Failed)"
            }

        prompt = self.build_prompt(query, retrieved)
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "你是一个基于检索增强生成的专业助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # 低温度保证一致性
            max_tokens=1024
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": [
                {
                    "doc_id": rd.document.metadata.get("parent_id"),
                    "chunk_index": rd.document.metadata.get("chunk_index"),
                    "score": rd.score
                }
                for rd in retrieved
            ],
            "pipeline": "RAG (Retrieval-Success)"
        }
```

### 企业部署架构

```
                        [负载均衡 Nginx / Traefik]
                                  |
                    +-------------+-------------+
                    |             |             |
              [API Gateway]  [API Gateway]  [API Gateway]
                    |             |             |
                    +-------------+-------------+
                                  |
                    +-------------+-------------+
                    |                           |
              [RAG Service]              [LLM Service]
              (检索 + 上下文)             (vLLM / TGI)
                    |                           |
              [向量数据库]                [GPU 集群]
              (Milvus/Qdrant)           (A100/H100)
                    |
              [Embedding Service]
              (SentenceTransformer)
                    |
              [文档处理 Pipeline]
              (OCR -> 解析 -> 分块 -> 嵌入)
```

### 性能优化策略

| 优化环节 | 策略 | 预期效果 |
|---------|------|---------|
| Embedding 生成 | 批量编码 + GPU 推理 | 10x 吞吐提升 |
| 向量检索 | IVF + PQ 索引 | 延迟 < 10ms |
| 上下文窗口 | Re-rank 后只取 Top-3 | 减少 40% token 消耗 |
| LLM 推理 | FlashAttention + vLLM | 3-5x 推理加速 |
| 缓存 | Redis 缓存热点查询 | 90% 缓存命中率 |
| 查询改写 | HyDE 假设文档生成 | 召回率提升 15% |

### 面试高频问题

**Q: RAG 与传统 QA 系统的本质区别是什么？**
> A: (1) RAG 利用 LLM 的理解和生成能力，而非预设模板；(2) RAG 支持多跳推理，而非单轮匹配；(3) RAG 的检索结果作为上下文注入，模型可以做信息综合、对比和推理；(4) 传统 QA 是"检索即答案"，RAG 是"检索后理解再生成"。

**Q: RAG 中的 chunking 策略对最终效果有多大影响？**
> A: 影响非常大。chunk 太小则上下文碎片化，太大则检索精度下降。实践中 256-512 tokens 适合段落级问答，1024-2048 tokens 适合长文档问答。关键原则是：chunk size 应与预期的问题粒度匹配，且 overlap 不应低于 10%。

---

## 1.7 混合检索 (Hybrid Search)

### 概念定义

混合检索是将**稀疏检索**（如 BM25，擅长精确匹配）和**稠密检索**（如 DPR/Embedding，擅长语义匹配）的结果融合，通过综合评分机制得到最终排序的技术。它弥补了单一检索方法在覆盖面与精确度上的各自不足。

### 为什么需要混合检索？

```
场景分析：

查询: "2024年Q3财报中的净利润"
                               BM25    Dense    混合检索
文档A: "2024年Q3净利润为120亿"   ✓       ✓        ✓ (双高)
文档B: "2024年第三季度利润情况"   ✗       ✓        ✓ (语义挽救)
文档C: "Q3净利润历史走势图"       ✓       ✗        ✓ (关键词挽救)
```

BM25 能精确匹配"Q3"和"净利润"，但可能漏掉"第三季度利润情况"。
稠密检索能匹配语义，但可能在长尾关键词上精度不足。
混合检索取长补短。

### 混合检索架构

```
用户查询
    |
    +---------------------+---------------------+
    |                     |                     |
    v                     v                     v
[查询改写 (Query Rewrite)]   [稀疏检索 (BM25)]      [稠密检索 (Dense)]
    |                     |                     |
    v                     v                     v
[HyDE 假设文档]            [Elasticsearch]        [向量数据库]
    |                     |                     |
    +---------+-----------+-------+-------------+
              |                   |
              v                   v
         [结果集 S]         [结果集 D]
              |                   |
              +---------+---------+
                        |
                        v
                   [融合排序 (Fusion)]
                        |
            +-----------+-----------+
            |                       |
            v                       v
      [线性加权融合]           [RRF (Reciprocal Rank Fusion)]
      score = α×BM25 + β×Dense    score = Σ 1/(k+rank)
                        |
                        v
                  [最终 Top-K]
```

### RRF (Reciprocal Rank Fusion)

RRF 是目前工业界最主流的融合算法：

```
RRF_score(d) = Σ_{r ∈ R} 1 / (k + rank_r(d))

其中：
- R 是所有检索器的集合
- rank_r(d) 是文档 d 在检索器 r 的结果列表中的排名
- k 是平滑常数，通常取 60
```

```python
from typing import List, Dict

class HybridSearchEngine:
    """混合检索引擎：BM25 + Dense + RRF"""

    def __init__(self, sparse_retriever, dense_retriever,
                 sparse_weight: float = 0.3, k_rrf: int = 60):
        self.sparse = sparse_retriever   # BM25 检索器
        self.dense = dense_retriever     # Dense 检索器
        self.w = sparse_weight
        self.k = k_rrf

    def linear_fusion(self, query: str, top_k: int = 10) -> List[dict]:
        """方法1: 线性加权融合"""
        sparse_results = self.sparse.search(query, top_k * 2)
        dense_results = self.dense.search(query, top_k * 2)

        # 归一化分数
        sparse_max = max(r.score for r in sparse_results) if sparse_results else 1
        dense_max = max(r.score for r in dense_results) if dense_results else 1

        combined_scores = {}
        for r in sparse_results:
            combined_scores[r.doc_id] = (
                self.w * (r.score / sparse_max),
                r.score / sparse_max,
                0
            )
        for r in dense_results:
            prev = combined_scores.get(r.doc_id, (0, 0, r.score / dense_max))
            combined_scores[r.doc_id] = (
                prev[0] + (1 - self.w) * (r.score / dense_max),
                prev[1],
                r.score / dense_max
            )

        ranked = sorted(combined_scores.items(),
                        key=lambda x: x[1][0], reverse=True)
        return ranked[:top_k]

    def rrf_fusion(self, query: str, top_k: int = 10) -> List[dict]:
        """方法2: RRF 融合 (更鲁棒，无需分数归一化)"""
        sparse_results = self.sparse.search(query, 100)
        dense_results = self.dense.search(query, 100)

        rrf_scores = {}
        for rank, result in enumerate(sparse_results):
            rrf_scores[result.doc_id] = 1 / (self.k + rank + 1)

        for rank, result in enumerate(dense_results):
            current = rrf_scores.get(result.doc_id, 0)
            rrf_scores[result.doc_id] = current + 1 / (self.k + rank + 1)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
```

### 融合策略对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| 线性加权 | 简单直观，可调节权重 | 需要分数归一化，对异常值敏感 | 检索器输出同量纲分数 |
| RRF | 不依赖绝对分数，鲁棒 | 忽略分数的量级信息 | 异构检索器融合 |
| 学习排序 (LTR) | 效果最优 | 需要标注数据和特征工程 | 高质量标注场景 |
| 级联融合 | 低延迟，先粗后精 | 粗筛阶段可能漏掉好结果 | 大规模实时检索 |

### 面试高频问题

**Q: 混合检索中如何确定 BM25 和 Dense 的权重？**
> A: (1) 离线评估：在验证集上网格搜索最佳权重；(2) 在线学习：通过用户点击反馈动态调整；(3) 查询分类：短查询/精确查询偏 BM25，长查询/语义查询偏 Dense；(4) 经验值：通常 BM25 权重 0.2-0.3 作为起始点。

---

## 1.8 GraphRAG

### 概念定义

GraphRAG (Graph + RAG) 是将**知识图谱 (Knowledge Graph)** 与 RAG 结合的高级检索范式。它通过知识图谱的结构化关系捕获实体之间的语义关联，使 RAG 不仅能检索语义相似的文本，还能沿着知识图谱的关系边进行图遍历，发现多跳关联和结构化知识。

### 微软 GraphRAG 架构 (2024)

```
[原始文档]
    |
    v
[实体抽取 & 关系抽取 (LLM)]
    |
    +-------------------+--------------------+
    |                   |                    |
    v                   v                    v
[实体节点]          [关系边]              [社区发现]
    |                   |                    |
    +-------------------+                    |
            |                                |
            v                                v
      [知识图谱]                      [社区摘要 (Community Summaries)]
            |                                |
            +---------------+----------------+
                            |
                            v
                   [图谱 + 文本混合索引]
                            |
                            v
                   [GraphRAG 检索层]
                            |
            +---------------+----------------+
            |                                |
            v                                v
    [局部检索 (Local)]            [全局检索 (Global)]
    - 实体关联展开                  - 社区摘要聚合
    - 邻居实体检索                  - 宏观知识问答
    - 文本 chunk 检索               - 概念层次查询
```

### GraphRAG 核心工作流

```python
class GraphRAGPipeline:
    """GraphRAG Pipeline 核心实现"""

    def extract_entities_relations(self, document: str, llm) -> dict:
        """Step 1 & 2: 使用 LLM 抽取实体和关系"""
        prompt = f"""请从以下文本中抽取实体和关系。

文本: {document}

请以 JSON 格式返回：
{{
  "entities": [
    {{"name": "...", "type": "PERSON/ORG/LOCATION/CONCEPT/...", "description": "..."}}
  ],
  "relations": [
    {{"source": "entity_name", "target": "entity_name", "relation": "关系描述"}}
  ]
}}"""
        return llm.generate_json(prompt)

    def build_knowledge_graph(self, documents: list) -> nx.Graph:
        """Step 3: 构建知识图谱"""
        import networkx as nx
        G = nx.Graph()

        for doc in documents:
            extracted = self.extract_entities_relations(doc.content)
            for entity in extracted['entities']:
                G.add_node(entity['name'], **entity)
            for rel in extracted['relations']:
                G.add_edge(rel['source'], rel['target'],
                          relation=rel['relation'], doc_id=doc.doc_id)
        return G

    def detect_communities(self, G: nx.Graph) -> dict:
        """Step 4: 社区发现 (Leiden 算法)"""
        from graspologic.partition import hierarchical_leiden
        community_map = hierarchical_leiden(
            G, max_cluster_size=10
        )
        return community_map

    def local_search(self, query: str, G: nx.Graph, top_k: int = 5) -> list:
        """局部检索: 实体 -> 邻居 -> 关联文本"""
        # 1. 提取查询中的实体
        query_entities = self.extract_entities_relations(query)
        # 2. 从实体出发进行 k-hop 邻居扩展
        context_entities = set()
        for entity in query_entities['entities']:
            if entity['name'] in G.nodes:
                # 获取 2-hop 邻居
                neighbors = nx.single_source_shortest_path_length(
                    G, entity['name'], cutoff=2
                )
                context_entities.update(neighbors.keys())
        # 3. 收集关联文本
        return self.collect_related_texts(context_entities)

    def global_search(self, query: str, communities: dict) -> list:
        """全局检索: 社区摘要匹配"""
        # 匹配与查询最相关的社区摘要
        return self.match_community_summaries(query, communities)
```

### GraphRAG 与传统 RAG 对比

```
场景: "苹果公司的供应链主要分布在哪些国家？"

传统 RAG (语义匹配):
   检索到: "苹果公司简介"、"苹果供应链报道"、"全球电子制造分布"
   -> 从多个独立 chunk 中综合信息
   -> 可能遗漏关键实体间的关系

GraphRAG (图谱遍历):
   苹果公司 --[拥有]--> 供应链网络
   供应链网络 --[分布在]--> 中国/TW
   供应链网络 --[分布在]--> 越南
   供应链网络 --[分布在]--> 印度
   供应链网络 --[分布在]--> 韩国
   -> 直接沿关系边遍历获取完整答案
```

### 技术选型矩阵

| 组件 | 方案 A | 方案 B | 方案 C |
|------|--------|--------|--------|
| 图数据库 | Neo4j | NebulaGraph | Neptune (AWS) |
| 实体抽取 | GPT-4 | GLiNER | SpaCy + LLM |
| 向量索引 | FAISS | pgvector | Elasticsearch |
| 图算法 | NetworkX | cuGraph (GPU) | igraph (R) |
| 社区发现 | Leiden | Louvain | Label Propagation |
| 开源方案 | Microsoft GraphRAG | Neo4j + LangChain | LlamaIndex + Nebula |

### 性能考量

| 阶段 | 瓶颈 | 优化方案 |
|------|------|---------|
| 实体抽取 | LLM 调用延迟 + 成本 | 批量处理 + 小模型初筛 |
| 图谱构建 | 大规模图的内存占用 | Neo4j 磁盘存储 + 缓存 |
| 图遍历 | 多跳遍历的指数爆炸 | 限制 hop 数 + 剪枝 |
| 语义检索 | 双路检索的延迟 | 并行化 + 预计算常用查询 |

### 面试高频问题

**Q: 什么场景下应该选择 GraphRAG 而非传统 RAG？**
> A: (1) 领域知识具有丰富的结构化关系（如医疗、法律、金融）；(2) 需要多跳推理才能回答的问题；(3) 需要全局视角（如"这个行业的整体竞争格局是什么？"）；(4) 对可解释性要求高的场景。如果只是简单的单跳事实检索，传统 RAG 性价比更高。

---

## 1.9 Agentic RAG

### 概念定义

Agentic RAG (Agent-driven RAG) 是将 **AI Agent** 与 RAG 结合，赋予 RAG 系统自主规划、工具使用、多步推理和自我纠错能力的范式。不同于传统 RAG 的"检索-生成"单次流水线，Agentic RAG 允许 Agent 根据生成质量动态决定是否需要补充检索、重写查询或调用外部工具。

### Agentic RAG 架构

```
+------------------------------------------------------------------+
|                        Agentic RAG System                         |
|                                                                   |
|  +-------------------+      +-------------------+                 |
|  |   Planner Agent   |----->|   Router Agent    |                |
|  | (任务分解 & 规划)   |      | (查询路由 & 分发)   |                |
|  +-------------------+      +--------+----------+                |
|                                      |                            |
|                   +------------------+------------------+         |
|                   |                  |                  |         |
|                   v                  v                  v         |
|          +--------+--------+ +------+-----+  +-------+------+   |
|          | Retriever Agent  | |Tool Agent  |  |Reflection    |   |
|          | (检索 & Rerank)  | |(计算器/API) |  |Agent (反思)  |   |
|          +--------+--------+ +------+-----+  +-------+------+   |
|                   |                  |                  |         |
|                   +------------------+------------------+         |
|                                      |                            |
|                                      v                            |
|                          +----------+---------+                  |
|                          | Synthesis Agent    |                  |
|                          | (综合 & 验证 & 输出)  |                 |
|                          +--------------------+                  |
+------------------------------------------------------------------+
```

### ReAct 模式 (Reasoning + Acting)

Agentic RAG 最常用的推理模式是 ReAct (Yao et al., 2022)：

```
Thought: 用户问"2024年相比2023年营收增长了百分之多少？"
         我需要先获取2024年和2023年的营收数据。
Action: retrieve("2024年度营收数据")
Observation: [检索结果] 2024年总营收为580亿元

Thought: 还需获取2023年的数据进行比较。
Action: retrieve("2023年度营收数据")
Observation: [检索结果] 2023年总营收为450亿元

Thought: 现在计算增长率：(580-450)/450 = 28.9%
Action: calculate("(580-450)/450*100")
Observation: 28.89

Thought: 我已经获得了完整答案，可以直接回复用户。
Answer: 2024年营收为580亿元，2023年为450亿元，同比增长约28.9%。
```

### 核心实现

```python
from typing import List, Dict, Any, Callable
import json
import re

class AgenticRAGSystem:
    """基于 ReAct 模式的 Agentic RAG"""

    def __init__(self, retriever, llm, tools: Dict[str, Callable]):
        self.retriever = retriever      # 检索引擎
        self.llm = llm                  # 大语言模型
        self.tools = tools              # 可用工具 {name: function}
        self.tools['retrieve'] = self.retrieve_wrapper
        self.max_steps = 10             # 最大推理步数

    def retrieve_wrapper(self, query: str) -> str:
        """检索工具封装"""
        results = self.retriever.retrieve(query)
        return json.dumps([{
            "content": r.document.content[:300],
            "score": r.score
        } for r in results], ensure_ascii=False)

    def build_react_prompt(self, query: str) -> str:
        """构建 ReAct 提示词"""
        tool_descriptions = "\n".join([
            f"- {name}: {func.__doc__ or 'No description'}"
            for name, func in self.tools.items()
        ])
        return f"""你是一个拥有工具调用能力的 AI Agent。请使用 ReAct 模式回答问题。

可用工具:
{tool_descriptions}

输出格式 (严格按以下格式):
Thought: [你对当前状态的思考]
Action: [工具名称]
Action Input: [工具输入参数]
Observation: [工具返回结果]
... (可重复)
Thought: 我现在有了足够信息
Final Answer: [最终回答]

用户问题: {query}"""

    def execute_step(self, thought: str, action: str, action_input: str) -> str:
        """执行单个工具调用步骤"""
        if action not in self.tools:
            return f"错误: 未知工具 '{action}'"
        try:
            result = self.tools[action](action_input)
            return str(result)
        except Exception as e:
            return f"工具执行错误: {str(e)}"

    def query(self, user_query: str) -> Dict[str, Any]:
        """Agentic RAG 查询入口"""
        prompt = self.build_react_prompt(user_query)
        messages = [{"role": "user", "content": prompt}]
        trajectory = []
        final_answer = None

        for step in range(self.max_steps):
            response = self.llm.chat(messages)
            response_text = response.content
            trajectory.append({"step": step, "response": response_text})

            # 检查是否到达最终答案
            if "Final Answer:" in response_text:
                final_answer = response_text.split("Final Answer:")[-1].strip()
                break

            # 解析 Action
            action_match = re.search(r"Action:\s*(.+)", response_text)
            input_match = re.search(r"Action Input:\s*(.+)", response_text)

            if action_match and input_match:
                action = action_match.group(1).strip()
                action_input = input_match.group(1).strip()
                observation = self.execute_step(
                    response_text, action, action_input
                )
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })
            else:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": "请继续推理，最终给出 Final Answer。"
                })

        return {
            "answer": final_answer or "未能完成推理",
            "trajectory": trajectory,
            "total_steps": len(trajectory)
        }
```

### 与传统 RAG 的对比

| 维度 | 传统 RAG | Agentic RAG |
|------|---------|-------------|
| 检索次数 | 单次检索 | 多次动态检索 |
| 决策逻辑 | 固定 Pipeline | Agent 自主决定 |
| 工具使用 | 仅检索 | 检索+计算+API+数据库 |
| 错误处理 | 无自纠错 | 可检测并重新检索 |
| 可解释性 | 低 | 高 (含推理轨迹) |
| 延迟 | 低 (1-3s) | 中 (5-20s) |
| 适合场景 | 简单事实问答 | 复杂的多跳分析问答 |

### 面试高频问题

**Q: 如何评估 Agentic RAG 的 Agent 决策质量？**
> A: (1) 轨迹分析：检查每一步的 Thought-Action-Observation 是否合理；(2) 工具调用准确率：Action 是否选择了最佳工具；(3) 步数效率：是否用最少的步数完成任务；(4) 终端成功率：最终答案是否正确；(5) 对比消融实验：去掉特定工具后评估效果变化。

---

## 1.10 Multi-Agent RAG

### 概念定义

Multi-Agent RAG (多智能体 RAG) 是将**多个专业化 AI Agent** 组成协作系统，每个 Agent 承担不同角色（如检索专家、领域专家、验证专家），通过消息传递和任务编排协同完成复杂 RAG 任务的架构范式。

### Multi-Agent RAG 架构

```
+------------------------------------------------------------------+
|                     Multi-Agent RAG System                        |
|                                                                   |
|                    +-------------------+                          |
|                    | Orchestrator Agent |                         |
|                    | (任务编排 & 调度)    |                         |
|                    +--------+----------+                          |
|                             |                                     |
|        +--------+-----------+-----------+---------+               |
|        |        |           |           |         |               |
|        v        v           v           v         v               |
|  +--------+ +--------+ +--------+ +--------+ +--------+         |
|  |Query   | |Search  | |Domain  | |Fact    | |Synthesis|         |
|  |Analyzer| |Agent   | |Expert  | |Checker | |Agent   |         |
|  +--------+ +--------+ +--------+ +--------+ +--------+         |
|      查询意图   多源检索   领域知识   事实验证   结果综合             |
|                                                                   |
|                    +-------------------+                          |
|                    |  Shared Memory    |                         |
|                    | (共享上下文 & 状态)  |                         |
|                    +-------------------+                          |
+------------------------------------------------------------------+
```

### 协作模式

```
模式 1: Pipeline (流水线)
QueryAnalyzer -> SearchAgent -> DomainExpert -> FactChecker -> SynthesisAgent

模式 2: Fan-out (广播式)
                    +- SearchAgent(Web) --------+
Orchestrator ------>+- SearchAgent(VectorDB) ----+-> SynthesisAgent
                    +- SearchAgent(KG) ---------+
                    +- SearchAgent(Structured) --+

模式 3: Debate (辩论式)
SearchAgent -> SynthesisAgent(A) -\
                                    >- JudgeAgent -> FinalAnswer
SearchAgent -> SynthesisAgent(B) -/

模式 4: Hierarchical (层级式)
LeadAgent
    |-> SubOrchestrator(DataRetrieval)
    |       |-> WebSearchAgent
    |       |-> VectorSearchAgent
    |       |-> GraphSearchAgent
    |-> SubOrchestrator(Analysis)
            |-> CalculatorAgent
            |-> StatisticalAgent
            |-> VisualizationAgent
```

### 角色定义与消息协议

```python
from typing import Any, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import json

class MessageType(Enum):
    TASK_ASSIGN = "task_assign"
    TASK_RESULT = "task_result"
    CONTEXT_QUERY = "context_query"
    CONTEXT_RESPONSE = "context_response"
    CORRECTION = "correction"
    FINAL_ANSWER = "final_answer"

@dataclass
class AgentMessage:
    """Agent 间消息协议"""
    msg_id: str
    msg_type: MessageType
    sender: str
    receiver: str
    content: Any
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    metadata: Dict = field(default_factory=dict)

@dataclass
class AgentRole:
    """Agent 角色定义"""
    name: str
    role: str
    system_prompt: str
    tools: List[str]
    output_schema: Dict
    dependencies: List[str]  # 依赖的其他 Agent

class MultiAgentRAGOrchestrator:
    """多 Agent RAG 编排器"""

    def __init__(self):
        self.agents: Dict[str, AgentRole] = {}
        self.message_queue: List[AgentMessage] = []
        self.shared_context: Dict[str, Any] = {}

    def register_agent(self, agent: AgentRole):
        """注册 Agent"""
        self.agents[agent.name] = agent

    def define_rag_team(self):
        """定义标准 RAG 团队"""
        self.register_agent(AgentRole(
            name="query_analyzer",
            role="查询分析专家",
            system_prompt="分析用户查询意图，分解为子查询，识别所需知识域",
            tools=["intent_classifier", "query_decomposer"],
            output_schema={"sub_queries": "list", "intent": "str"},
            dependencies=[]
        ))
        self.register_agent(AgentRole(
            name="search_agent",
            role="搜索专家",
            system_prompt="执行多源搜索：向量库、ES、Web、结构化DB",
            tools=["vector_search", "es_search", "web_search"],
            output_schema={"documents": "list", "sources": "list"},
            dependencies=["query_analyzer"]
        ))
        self.register_agent(AgentRole(
            name="domain_expert",
            role="领域专家",
            system_prompt="结合领域知识库验证和补充检索结果",
            tools=["domain_kb_query", "terminology_check"],
            output_schema={"verified_docs": "list", "insights": "list"},
            dependencies=["search_agent"]
        ))
        self.register_agent(AgentRole(
            name="fact_checker",
            role="事实验证专家",
            system_prompt="验证事实准确性，标注不确定项",
            tools=["cross_reference", "confidence_scorer"],
            output_schema={"verified": "bool", "issues": "list"},
            dependencies=["domain_expert"]
        ))
        self.register_agent(AgentRole(
            name="synthesis_agent",
            role="综合生成专家",
            system_prompt="综合所有结果，生成准确、完整、有引用的回答",
            tools=["citation_formatter", "quality_scorer"],
            output_schema={"answer": "str", "citations": "list"},
            dependencies=["fact_checker"]
        ))

    def execute_pipeline(self, user_query: str) -> Dict:
        """执行 Pipeline 模式的 Multi-Agent RAG"""
        results = {}
        current_context = {"original_query": user_query}

        execution_order = [
            "query_analyzer", "search_agent",
            "domain_expert", "fact_checker", "synthesis_agent"
        ]

        for agent_name in execution_order:
            agent = self.agents[agent_name]
            # 检查依赖是否已完成
            deps_ready = all(
                dep in results for dep in agent.dependencies
            )
            if not deps_ready:
                raise RuntimeError(
                    f"Agent '{agent_name}' 依赖未满足: {agent.dependencies}"
                )

            # 构建当前 Agent 的输入上下文
            agent_input = {
                "current_context": current_context,
                "previous_results": {
                    dep: results[dep] for dep in agent.dependencies
                },
                "system_prompt": agent.system_prompt
            }

            # 模拟 Agent 执行 (实际应为 LLM 调用)
            agent_output = self.invoke_agent(agent, agent_input)
            results[agent_name] = agent_output
            self.shared_context[agent_name] = agent_output

            # 更新上下文
            current_context.update(agent_output.get("context_update", {}))

        return {
            "final_answer": results["synthesis_agent"].get("answer"),
            "trace": results,
            "shared_context": dict(self.shared_context)
        }

    def invoke_agent(self, agent: AgentRole, agent_input: Dict) -> Dict:
        """调用 LLM 执行 Agent 任务"""
        prompt = f"""{agent.system_prompt}

当前上下文: {json.dumps(agent_input['current_context'], ensure_ascii=False)}
前置结果: {json.dumps(agent_input['previous_results'], ensure_ascii=False)}

请按以下格式输出: {json.dumps(agent.output_schema, ensure_ascii=False)}
"""
        # 实际调用 LLM
        # response = llm_client.chat(prompt)
        # return parse_output(response, agent.output_schema)
        return {"placeholder": f"{agent.name} 的执行结果"}
```

### 多 Agent 编排对比

| 模式 | 延迟 | 并行度 | 鲁棒性 | 成本 | 适用场景 |
|------|------|--------|--------|------|---------|
| Pipeline | 低 | 串行 | 单点脆弱 | 低 | 标准 RAG |
| Fan-out | 中 | 高 | 高 | 中 | 多源综合 |
| Debate | 高 | 中 | 最高 | 高 | 高风险决策 |
| Hierarchical | 中 | 中高 | 高 | 中 | 复杂分解任务 |
| Swarm (Mesh) | 中 | 最高 | 最高 | 高 | 大规模开放问题 |

### 面试高频问题

**Q: Multi-Agent RAG 相比单 Agent RAG 的核心优势是什么？**
> A: (1) 专业化分工：每个 Agent 只需精通小领域，降低单个 Agent 的复杂度；(2) 并行执行：多源检索可并发进行，降低端到端延迟；(3) 容错性：单个 Agent 失败不会导致全部失败；(4) 质量保证：专职 Fact Checker 可捕获检索幻觉；(5) 可扩展性：可按需增加新 Agent 扩展能力。但代价是系统复杂度和 Token 消耗增加。

---

## 深度专题

### 专题一：为什么 LLM 需要 RAG？

```
LLM 的三大原生缺陷：

1. 知识截止 (Knowledge Cutoff)
   GPT-4 训练数据截止 2023年10月
   问题: "2024年的美国总统是谁？"
   LLM 内部知识: 无相关信息
   RAG 解决方案: 检索 2024 年新闻 -> 正确回答

2. 幻觉 (Hallucination)
   LLM 倾向于生成看似合理但实际错误的内容
   问题: "请引用《民法典》第1043条原文"
   LLM 内部知识: 可能编造法条
   RAG 解决方案: 检索《民法典》原文 -> 精确引用

3. 知识密度 (Knowledge Density)
   LLM 参数化知识受限于参数量，对长尾知识覆盖不足
   问题: "xx公司2023年Q2财报中的研发费用是多少？"
   LLM 内部知识: 不可能存储所有公司的财报数据
   RAG 解决方案: 检索企业知识库 -> 精准数值
```

**知识更新频率对比**：

| 方法 | 知识更新周期 | 成本 | 效果 |
|------|------------|------|------|
| 重新训练 | 数月 | 数百万美元 | 全面更新 |
| 微调 (Fine-tuning) | 数小时-数天 | 数百-数千美元 | 局部更新 |
| RAG | 实时 (秒级) | < 0.01 美元/查询 | 按需检索 |

### 专题二：为什么 Fine-tuning 无法独立解决知识更新问题？

```
+------------------------------------------------------------------+
|              知识更新的本质性矛盾                                 |
+------------------------------------------------------------------+

1. 参数化知识的局限
   - 模型参数是固定的浮点数矩阵
   - 通过梯度下降修改这些参数 = 改变"神经元"的连接强度
   - 问题: 这是一个全局优化过程,无法保证增量更新不破坏已有知识

2. 灾难性遗忘 (Catastrophic Forgetting)
   - Fine-tune 新知识 -> 旧知识的决策边界被扰动
   - 解法 (EWC, Replay) 增加了复杂度但无法完全消除

3. 时效性悖论
   - 训练需要数小时到数天
   - 新信息每分钟都在产生
   - Fine-tuning 永远追不上信息产生的速度

4. 事实更新的不可控性
   - 你无法精确地"只修改模型对'拜登是总统'的认知"
   - 修改任何参数都会产生蝴蝶效应
   - 对比: RAG 中直接删除/更新向量索引中的对应文档即可
```

**对比实验数据**：

| 指标 | Fine-tuning | RAG | RAG + Fine-tuning |
|------|------------|-----|-------------------|
| 知识更新延迟 | 小时级 | 秒级 | 秒级 |
| 单次更新成本 | $10-1000 | $0.001 | $10-1000 |
| 事实准确率 | 68% | 89% | 92% |
| 旧知识保持率 | 72% | 100% | 85% |
| 幻觉率 | 15% | 4% | 3% |
| 可解释性 | 低 | 高 | 中高 |

### 专题三：RAG 与 Fine-tuning 的本质区别

```
+---------------------+-----------------------------+----------------------------+
| 维度                 | RAG                         | Fine-tuning                |
+---------------------+-----------------------------+----------------------------+
| 知识存储方式          | 外部知识库 (向量数据库)      | 模型参数内部                |
| 知识获取方式          | 检索 (Retrieve)             | 梯度下降优化 (Optimize)     |
| 知识类型              | 显性知识 (Explicit)         | 隐性知识 (Implicit)         |
| 更新粒度              | 文档级别 (精确可控)          | 参数级别 (难以精确控制)      |
| 更新速度              | 实时 (秒)                   | 慢 (小时-天)                |
| 知识边界              | 清晰 (可审计)               | 模糊 (黑盒)                 |
| 推理方式              | 上下文推理 (In-context)     | 参数推理 (Parametric)       |
| 成本结构              | 检索成本 + LLM 推理成本     | 训练成本 (一次性) + 推理成本 |
| 适配新领域要求        | 更新知识库即可              | 需要训练数据 + GPU 资源     |
| 可解释性              | 高 (可追溯每条知识的来源)    | 低 (知识如何存储不可解释)    |
+---------------------+-----------------------------+----------------------------+
```

**架构视角的本质区别**：

```
Fine-tuning 模型:
  [输入] -> [模型参数 (含新知识)] -> [输出]
  知识 = 参数的一部分

RAG 模型:
  [输入] -> [检索器] -> [知识库] -> [上下文] -+
                                              +-> [LLM 推理] -> [输出]
  [输入] -----------------------------------+

  知识 = 外部可替换资源
```

**核心结论**：
- Fine-tuning 改变了**模型本身** -- 适用于风格适配、格式控制、行为塑造
- RAG 改变了**模型的输入** -- 适用于事实注入、知识更新、内容溯源
- **两者互补而非对立**：Fine-tuning 让模型学会"如何使用检索到的信息"，RAG 提供最新的外部知识

---

## 全章总结

### 技术演进总览

```
         +-----------+     +-----------+     +-----------+
1970s -> | 传统搜索   | --> | 全文检索   | --> | ES        |
         | (IR/VSM)  |     | (倒排/TF-IDF)|   | (分布式)   |
         +-----------+     +-----------+     +-----------+
                                                    |
         +-----------+     +-----------+            |
2018 -> | 向量检索   | --> | DPR/稠密   | <---------+
         | (ANN)     |     | (BERT)    |
         +-----------+     +-----------+
                                |
         +-----------+         |
2020 -> | RAG       | <-------+
         | (Lewis)   |
         +-----------+
              |
         +-----------+     +-----------+     +-----------+
2022 -> | 混合检索   | --> | GraphRAG  | --> | Agentic   |
         | (RRF)     |     | (KG+RAG)  |     | RAG       |
         +-----------+     +-----------+     +-----------+
                                                    |
                                               +-----------+
                                   2024 ->     | Multi-    |
                                               | Agent RAG |
                                               +-----------+
```

### 企业 RAG 选型决策树

```
                     开始
                      |
                是否需要实时外部知识？
                /              \
              否                 是
              |                  |
         直接用 LLM          需要结构化推理？
         (或 Fine-tune)      /         \
                           否            是
                           |             |
                      文档量大吗？    有现成知识图谱？
                      /       \        /        \
                    <100万   >100万    有         没有
                     |        |        |          |
                 简单 RAG   混合检索  GraphRAG   先构建KG
                             + RRF              再用GraphRAG
                 |
           是否需要多步推理？
           /              \
         否                 是
         |                  |
     单次 RAG           需要多工具？
     Pipeline           /         \
                      否           是
                      |            |
                 Agentic RAG   Multi-Agent RAG
                 (ReAct)       (Pipeline/Fan-out)
```

### 面试速查卡

| 技术 | 核心公式/算法 | 一句话优势 | 主要局限 |
|------|-------------|-----------|---------|
| 传统搜索 | VSM, 概率模型 | 成熟稳定 | 无排序能力 |
| 全文检索 | TF-IDF, 倒排索引 | 精确匹配 | 无语义理解 |
| ES | BM25, Analyzer | 分布式, 生态完善 | 稀疏向量局限 |
| 向量检索 | ANN (HNSW/IVF) | 语义匹配 | 需训练模型 |
| 稠密检索 | DPR, 双编码器 | 深度语义 | 训练数据需求 |
| RAG | Retrieve + Generate | 实时知识注入 | 检索质量依赖 |
| 混合检索 | RRF 融合 | 精准+全面 | 系统复杂度 |
| GraphRAG | KG + 社区发现 | 结构推理 | 构建成本高 |
| Agentic RAG | ReAct, Tool-use | 自主决策 | 延迟较高 |
| Multi-Agent RAG | Pipeline/Fan-out | 专业化协作 | 编排复杂 |

---

> **下一章预告**：第二章将深入 RAG 核心架构设计，涵盖 Chunking 策略选型、Embedding 模型评测体系、向量数据库对比与选型、Rerank 后处理、以及端到端性能调优方法论。
