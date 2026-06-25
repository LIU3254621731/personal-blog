# 第八章：Rerank 重排序系统

> **章节目录**：[8.1 为什么仅靠 TopK 召回不够？](#81-为什么仅靠-topk-召回不够) | [8.2 Rerank 架构深度解析](#82-rerank-架构深度解析) | [8.3 主流 Rerank 模型对比](#83-主流-rerank-模型对比) | [8.4 Rerank 流水线实现](#84-rerank-流水线实现) | [8.5 高级 Rerank 技术](#85-高级-rerank-技术) | [8.6 企业级部署](#86-企业级部署)

---

## 8.1 为什么仅靠 TopK 召回不够？

### 8.1.1 语义鸿沟：Embedding 相似度不等于相关性

在 RAG 系统中，第一阶段召回（Recall）通常依赖向量相似度（如余弦相似度）从海量文档中检索 TopK 候选。然而，**向量相似度高并不代表文档真正回答了用户的问题**。这种现象被称为"语义鸿沟"（Semantic Gap），其根源在于：

**Embedding 模型的本质限制**：现代 Sentence Embedding 模型（如 text-embedding-3-large、bge-large-zh）采用 Bi-Encoder（双塔）架构，Query 和 Document 分别独立编码为固定维度的稠密向量，之后通过向量点积/余弦距离计算相似度。在此过程中，**Query 和 Document 之间从未发生逐 Token 的深度交互**——它们在嵌入空间中是两条平行计算、最后才通过一个标量值（相似度得分）建立联系的向量。

```
+------------------------------------------------------------------+
|                    Bi-Encoder 架构（召回阶段）                      |
+------------------------------------------------------------------+
|                                                                    |
|    Query: "如何配置Redis集群?"        Document: "Redis集群部署指南"   |
|         |                                     |                   |
|         v                                     v                   |
|   [Query Encoder]                       [Doc Encoder]             |
|         |                                     |                   |
|         v                                     v                   |
|   q = [0.12, -0.34, ...]             d = [0.09, -0.41, ...]      |
|         |                                     |                   |
|         +---------- 余弦(q, d) = 0.87 ---------+                   |
|                                                                    |
|   X 信息瓶颈：两个独立向量的一次点积，丢失了 Token 级交互信息 X       |
+------------------------------------------------------------------+
```

**为什么 Embedding 相似度会产生误导？** 考虑以下场景：

| 用户问题 | 召回文档 | 向量相似度 | 是否相关? | 原因 |
|---------|---------|-----------|----------|------|
| "Redis 集群最少需要几个节点？" | "Redis 集群最少 6 个节点（3主3从）" | 0.91 | 相关 | 精确匹配 |
| "Redis 集群最少需要几个节点？" | "Redis 集群最多支持16384个槽位" | 0.85 | **不相关** | 都涉及"Redis集群"但答案不同 |
| "Redis 集群最少需要几个节点？" | "MySQL 集群最少需要3个节点" | 0.79 | **不相关** | "集群"+"最少"+"节点"词共现 |
| "Redis 集群最少需要几个节点？" | "为何Redis 集群选用奇数节点以达成Quorum" | 0.88 | 部分相关 | 涉及节点数但回答的是"为何" |

向量相似度将 Query 压缩为一个"平均语义方向"，无法分辨：
- **词序**："A 打败了 B" vs "B 打败了 A" 在 Embedding 空间可能非常接近
- **否定**："不支持事务" 与 "支持事务" 只有一字之差，Embedding 难以区分
- **精确约束**："最少"、"必须"、">=100" 这类硬约束在向量空间中天生被"软化"
- **细粒度匹配**：Query 中的"配置"一词是否与文档中的"配置文件"在 Token 级精确对齐——Bi-Encoder 无从得知

**因此，我们需要 Rerank（重排序）：在粗召回之后，用更强的模型对候选集进行精细排序，弥合语义鸿沟。**

### 8.1.2 Rerank 在 RAG 流水线中的定位

```
+------------------------------------------------------------------+
|                RAG 检索流水线（两阶段架构）                         |
+------------------------------------------------------------------+
|                                                                    |
|  [用户Query]                                                        |
|       |                                                             |
|       v                                                             |
|  +------------+    +------------+    +------------+                 |
|  | 查询重写    | -> | 多路召回    | -> | Rerank     | -> [Top N]    |
|  | (Query     |    | (混合检索)  |    | (精细排序)  |    最终结果    |
|  |  Rewrite)  |    | K=100~200  |    | N=5~10     |                |
|  +------------+    +------------+    +------------+                 |
|                        |                  |                         |
|                   向量 + BM25         Cross Encoder                 |
|                   关键词 + 知识图谱   ColBERT / LLM                  |
|                                                                    |
|  阶段:   前置处理         粗召回              精排                    |
|  目标:   理解意图         高召回率            高精确率                 |
|  延迟:   <50ms           <200ms             <500ms                  |
|  模型:   规则/小模型       Bi-Encoder        Cross Encoder           |
+------------------------------------------------------------------+
```

召回阶段追求**高召回率**（Recall@K → 100%），允许噪声存在；Rerank 阶段追求**高精确率**（Precision@N），从噪声中筛选精华。这种"宽进严出"的策略是当前工业界 RAG 系统的标准范式。

---

## 8.2 Rerank 架构深度解析

### 8.2.1 Cross Encoder（交叉编码器）

#### 工作原理

Cross Encoder 将 Query 和 Document **拼接为单一序列**送入 Transformer，在每一层 Self-Attention 中 Query 的每个 Token 都能直接关注 Document 的每个 Token，实现**全对全的 Token 级交互**。

```
+------------------------------------------------------------------+
|                    Cross Encoder 架构                              |
+------------------------------------------------------------------+
|                                                                    |
|  Input: [CLS] query_tok_1 ... query_tok_m [SEP] doc_tok_1 ... [SEP]|
|                                                                    |
|          +--------------------------------------------------+     |
|          |              Transformer Layers                   |     |
|          |                                                  |     |
|          |   Self-Attention: 每个 Token 关注所有 Token       |     |
|          |                                                  |     |
|          |   Query Tokens ←→ Doc Tokens (全对全交互)         |     |
|          |   Query Tokens ←→ Query Tokens (自我交互)         |     |
|          |   Doc Tokens ←→ Doc Tokens (上下文交互)           |     |
|          |                                                  |     |
|          +--------------------------------------------------+     |
|                              |                                     |
|                              v                                     |
|               [CLS] → Linear → Sigmoid → Relevance Score (0~1)    |
|                                                                    |
+------------------------------------------------------------------+
```

#### 为什么比 Bi-Encoder 更精确？

| 维度 | Bi-Encoder | Cross Encoder |
|-----|-----------|---------------|
| **交互方式** | 编码后点积（late interaction） | 编码中全注意力（early full interaction） |
| **Token 级匹配** | 不支持（Query/Doc 独立编码） | 支持（Self-Attention over concatenated input） |
| **词序感知** | 间接（位置编码融入 Embedding） | 直接（Attention 能看到完整词序） |
| **否定词处理** | 弱（"不支持" vs "支持"向量接近） | 强（"不"和"支持"在 Attention 中建立否定关系） |
| **推理速度** | 快（Doc 向量可预计算） | 慢（每次 Query 需重新编码所有候选 Doc） |
| **存储开销** | Doc 向量需存储（~1.5KB/doc） | 无需预存 Doc 向量 |
| **适用场景** | 召回（百万级 TopK） | 精排（数十至数百级 TopK） |

#### Python 实现示例

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Tuple

class CrossEncoderReranker:
    """Cross Encoder 重排序器"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", 
                 max_length: int = 8192, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.max_length = max_length
        self.device = device
        
    @torch.no_grad()
    def rerank(self, query: str, documents: List[str], 
               top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        对文档列表进行重排序
        
        参数:
            query: 用户查询
            documents: 候选文档列表
            top_k: 返回的文档数量
            
        返回:
            List of (index, score, document) 按分数降序排列
        """
        # 构建 [query, doc] 对
        pairs = [[query, doc] for doc in documents]
        
        # 分批编码以避免 OOM
        batch_size = 32
        all_scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu()
            all_scores.extend(scores.tolist())
        
        # 排序并返回 TopK
        idx_score_doc = [(i, score, doc) 
                         for i, (score, doc) in enumerate(zip(all_scores, documents))]
        idx_score_doc.sort(key=lambda x: x[1], reverse=True)
        
        return idx_score_doc[:top_k]


# 使用示例
reranker = CrossEncoderReranker()
query = "Redis 集群最少需要几个节点？"
candidates = [
    "Redis 集群最少需要 6 个节点，包括 3 个主节点和 3 个从节点。",
    "Redis 集群最多支持 16384 个哈希槽位。",
    "MySQL 集群最少需要 3 个节点即可实现高可用。",
    "Redis 哨兵模式最少需要 3 个节点来进行故障转移。",
]

results = reranker.rerank(query, candidates, top_k=3)
for idx, score, doc in results:
    print(f"[{score:.4f}] {doc[:80]}...")
```

### 8.2.2 Bi-Encoder（双塔编码器）

#### 工作原理

Bi-Encoder 使用两个独立（或共享权重）的 Encoder 分别编码 Query 和 Document，然后通过向量相似度函数（点积/余弦）计算相关性。这是召回阶段的默认架构。

```
+------------------------------------------------------------------+
|                    Bi-Encoder 架构                                 |
+------------------------------------------------------------------+
|                                                                    |
|   Query: "Redis集群配置"           Document: "Redis集群部署指南..."  |
|        |                                   |                       |
|        v                                   v                       |
|  [Query Tower]                        [Document Tower]            |
|  (Transformer)                        (Transformer)               |
|        |                                   |                       |
|        v                                   v                       |
|    q ∈ R^d                             d ∈ R^d                    |
|        |                                   |                       |
|        +------ 相似度(q, d) = q·d  --------+                       |
|                  (仅此一次交互！)                                    |
|                                                                    |
|  特征:                                                             |
|  - Document 向量可离线预计算和索引 → 召回速度极快                     |
|  - Query 和 Doc 之间的 Token 级交互 = 0                             |
|  - 信息瓶颈: d 维向量承载的语义容量有限                              |
+------------------------------------------------------------------+
```

#### 核心矛盾：效率 vs 精度

Bi-Encoder 的精度上限受限于**Pooling 导致的信息损失**。无论采用 Mean Pooling、[CLS] Pooling 还是 Last-Token Pooling，将变长文档压缩为固定维度向量必然丢失细粒度信息。在 MS MARCO 等基准上，同等规模的 Cross Encoder 通常比 Bi-Encoder 的 MRR@10 高出 3-8 个百分点。

### 8.2.3 ColBERT（Late Interaction / 延迟交互）

#### 工作原理

ColBERT（Contextualized Late Interaction over BERT）是一种**介于 Bi-Encoder 和 Cross Encoder 之间的折中方案**。它保留 Query 和 Document 各自的 Token 级表示（不再 Pooling 为单向量），在检索时计算 **MaxSim（Maximum Similarity）操作**——Query 的每个 Token 找到 Document 中最相似的 Token，取最大值后求和。

```
+------------------------------------------------------------------+
|                    ColBERT 架构 (Late Interaction)                 |
+------------------------------------------------------------------+
|                                                                    |
|   Query: "Redis 集群 最少 节点"                                     |
|                                                                    |
|   Query Encoder → Q = [q1, q2, q3, q4]    (4 tokens × d dims)     |
|                                                                    |
|   Document: "Redis cluster requires at least 6 nodes..."           |
|                                                                    |
|   Doc Encoder → D = [d1, d2, d3, ..., dk]   (k tokens × d dims)   |
|                                                                    |
|   +-----------------------------------------------------------+   |
|   |  MaxSim 计算：                                             |   |
|   |                                                           |   |
|   |  对于 Query 中的每个 Token q_i:                            |   |
|   |    score_i = max_{j=1..k} (q_i · d_j)    ← 找最匹配的Doc Token|
|   |                                                           |   |
|   |  最终得分 = Σ_{i=1}^{|Q|} score_i                          |   |
|   |                                                           |   |
|   |  示例:                                                    |   |
|   |  q_"Redis" → max over Doc tokens → 匹配 "Redis" → 0.95    |   |
|   |  q_"最少"  → max over Doc tokens → 匹配 "at least" → 0.82 |   |
|   |  q_"节点"  → max over Doc tokens → 匹配 "nodes" → 0.91    |   |
|   |  q_"集群"  → max over Doc tokens → 匹配 "cluster" → 0.88  |   |
|   |                                                           |   |
|   |  Sum = 0.95 + 0.82 + 0.91 + 0.88 = 3.56                   |   |
|   +-----------------------------------------------------------+   |
|                                                                    |
+------------------------------------------------------------------+
```

#### ColBERT vs Cross Encoder vs Bi-Encoder

```
+------------------------------------------------------------------+
|                  三种架构全面对比                                   |
+------------------------------------------------------------------+
|                                                                    |
|  维度              Bi-Encoder    ColBERT         Cross Encoder     |
|  ─────────────────────────────────────────────────────────────     |
|  Query-Doc交互     编码后点积     Token级MaxSim   全对全注意力       |
|  Token级匹配       无            有(每个Q Token)  有(所有Token)     |
|  交互复杂度         O(1)          O(|Q|×|D|)      O((|Q|+|D|)^2)  |
|  文档向量化         是(单向量)     是(多向量)       否                |
|  文档存储/文档      1.5KB         25KB×|D|        0                |
|  推理速度          极快          快              慢                |
|  精度(MRAR@10)     ~38%          ~42%            ~45%             |
|  召回阶段适用      核心方案       可选            不适用            |
|  Rerank阶段适用    不适用         适用(轻量)      适用(主力)        |
|                                                                    |
+------------------------------------------------------------------+
```

**MaxSim 的数学定义**：

```
ColBERT_Score(q, d) = Σ_{i=1}^{|q|} max_{j=1}^{|d|} (E_q(q_i) · E_d(d_j)^T)
```

其中：
- `E_q(q_i)` 是 Query 第 i 个 Token 的上下文嵌入向量（经 L2 归一化）
- `E_d(d_j)` 是 Document 第 j 个 Token 的上下文嵌入向量（经 L2 归一化）
- `max_{j}` 是 Document 侧对当前 Query Token 的最大相似度匹配

---

## 8.3 主流 Rerank 模型对比

### 8.3.1 模型详情

#### BGE Reranker（BAAI）

由北京智源人工智能研究院（BAAI）开发，是中文 RAG 生态中最广泛使用的 Reranker 系列。

| 模型 | 参数量 | 最大长度 | 语言 | 特点 |
|------|--------|---------|------|------|
| bge-reranker-base | 278M | 512 | 中/英 | 轻量，速度快 |
| bge-reranker-large | 560M | 512 | 中/英 | 精度更高 |
| bge-reranker-v2-m3 | 568M | 8192 | 多语言 | 支持长文本，多语言 |
| bge-reranker-v2-gemma | 2.2B | 8192 | 多语言 | Gemma 骨干，精度最高 |
| bge-reranker-v2-minicpm | 2.4B | 8192 | 多语言 | MiniCPM 骨干，速度快 |

**核心设计**：
- 基于 XLM-RoBERTa 架构，以 [CLS] Token 的输出做二分类（相关/不相关）
- 训练数据：多阶段训练，包括 MS MARCO、NQ、DuReader 等
- 支持 Instruction-aware Reranking（v2 系列）

#### Qwen Reranker（阿里巴巴）

| 模型 | 参数量 | 最大长度 | 语言 | 特点 |
|------|--------|---------|------|------|
| gte-qwen2-1.5b-instruct | 1.5B | 32K | 多语言 | 基于 Qwen2 架构 |
| gte-qwen2-7b-instruct | 7B | 32K | 多语言 | 精度极高，需更多资源 |

**核心设计**：
- 基于 Qwen2 Decoder-only 架构，使用 Last Token 输出做相关性判断
- 支持超长上下文（32K tokens），适合长文档重排序
- Instruction-tuned，可通过 Prompt 调整排序行为

#### Jina Reranker

| 模型 | 参数量 | 最大长度 | 语言 | 特点 |
|------|--------|---------|------|------|
| jina-reranker-v1-base-en | 137M | 512 | 英文 | 极轻量 |
| jina-reranker-v2-base-multilingual | 278M | 8192 | 多语言(100+) | 轻量多语言方案 |
| jina-reranker-v3 | 568M | 8192 | 多语言 | 支持 Task-specific LoRA |

**核心设计**：
- jina-reranker-v3 引入 Task-specific LoRA Adapters，同一模型通过不同 LoRA 权重适配不同任务（Question-Answering、Retrieval、Code-Search 等）
- Flash Attention 优化推理速度

#### Cohere Reranker

Cohere 的 Rerank 服务是闭源 API，不需要自行部署模型。

| 版本 | 最大长度 | 语言 | 特点 |
|------|---------|------|------|
| rerank-english-v3.0 | 4096 | 英文 | 英文精排专用 |
| rerank-multilingual-v3.0 | 4096 | 多语言(100+) | 多语言 Rerank |
| rerank-v3.5 | 8192 | 多语言 | 更长上下文，更高精度 |

**核心设计**：
- 专有模型，细节未完全公开
- API 调用模式，按搜索请求计费
- 内置 Token 级交互优化

### 8.3.2 综合对比表

```
+------------------------------------------------------------------------------------------------------+
|                              Rerank 模型综合对比 (2025 Q2 数据)                                         |
+------------------------------------------------------------------------------------------------------+
|                                                                                                      |
| 模型                     | NDCG@10  | 速度      | GPU显存  | 支持语言   | 最大长度 | 部署难度 | 成本    |
|                          | (BEIR)   | (q/s)     | (FP16)   |           |         |         |        |
| ─────────────────────────┼──────────┼───────────┼──────────┼───────────┼─────────┼─────────┼────────|
| bge-reranker-base        | 48.2     | ~120      | ~550MB   | 中/英     | 512     | 低      | 免费    |
| bge-reranker-large       | 50.1     | ~80       | ~1.1GB   | 中/英     | 512     | 低      | 免费    |
| bge-reranker-v2-m3       | 52.7     | ~65       | ~1.1GB   | 多语言    | 8192    | 低      | 免费    |
| bge-reranker-v2-gemma    | 55.3     | ~30       | ~4.4GB   | 多语言    | 8192    | 中      | 免费    |
| bge-reranker-v2-minicpm  | 54.8     | ~40       | ~4.8GB   | 多语言    | 8192    | 中      | 免费    |
| gte-qwen2-1.5b-instruct  | 54.1     | ~45       | ~3.0GB   | 多语言    | 32K     | 中      | 免费    |
| gte-qwen2-7b-instruct    | 56.8     | ~15       | ~14GB    | 多语言    | 32K     | 高      | 免费    |
| jina-reranker-v2-multi   | 50.5     | ~100      | ~550MB   | 多语言    | 8192    | 低      | 免费    |
| jina-reranker-v3         | 52.1     | ~70       | ~1.1GB   | 多语言    | 8192    | 低      | 免费    |
| cohere-rerank-v3.5       | 55.9     | API依赖   | N/A(API) | 多语言    | 8192    | 无需部署 | $2/1M次 |
| ─────────────────────────┴──────────┴───────────┴──────────┴───────────┴─────────┴─────────┴────────|
|                                                                                                      |
|  注意: NDCG@10 数据来自 MTEB/BEIR 基准的检索子任务，实际场景差异较大                                    |
|  速度(q/s)为 A100 80GB 上 batch_size=32 的近似值，实际受文档长度和批次大小影响                          |
|                                                                                                      |
+------------------------------------------------------------------------------------------------------+
```

### 8.3.3 技术选型建议

```
+------------------------------------------------------------------+
|                    Rerank 模型选型决策树                            |
+------------------------------------------------------------------+
|                                                                    |
|  [需要中文支持?]                                                    |
|       |                                                           |
|   是  |  否                                                        |
|       v                                                           |
|  [GPU 显存 >= 8GB?]          [需要 API 免部署?]                     |
|       |                           |                               |
|   是  |  否                   是   |  否                           |
|       v                           v                               |
|  [对精度要求极致?]         Cohere Rerank API    [bge-reranker-v2-  |
|       |                      (多语言)              m3 / jina-v3]   |
|   是  |  否                                                     |
|       v                                                         |
|  gte-qwen2-7b    [bge-reranker-v2-m3]                            |
|  (精度第一)       (性价比最佳)                                    |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 8.4 Rerank 流水线实现

### 8.4.1 两阶段检索架构（核心范式）

```
+------------------------------------------------------------------+
|                    两阶段检索流水线（完整数据流）                    |
+------------------------------------------------------------------+
|                                                                    |
|  INPUT: user_query = "Redis集群最少节点数"                          |
|                                                                    |
|  ┌─────────────────────────────────────────────────────────────┐  |
|  │ 阶段1: 快速召回 (Recall Phase)             延迟预算: 200ms    │  |
|  │                                                             │  |
|  │   Query → [向量检索] → FAISS Top-100     延迟: ~50ms         │  |
|  │   Query → [BM25检索] → Elasticsearch Top-100 延迟: ~30ms    │  |
|  │   Query → [关键词检索] → 倒排索引 Top-50   延迟: ~20ms       │  |
|  │                         |                                    │  |
|  │                         v                                    │  |
|  │               RRF 融合 → 去重 → Top-200                       │  |
|  │                         |                                    │  |
|  └─────────────────────────|────────────────────────────────────┘  |
|                            v                                       |
|  ┌─────────────────────────────────────────────────────────────┐  |
|  │ 阶段2: 精细重排 (Rerank Phase)             延迟预算: 500ms    │  |
|  │                                                             │  |
|  │   Top-200 候选 → [Cross Encoder 打分] → 排序 → Top-10        │  |
|  │                         |                                    │  |
|  │                         v                                    │  |
|  │               [MMR 多样性优化] → Top-5                        │  |
|  │                         |                                    │  |
|  └─────────────────────────|────────────────────────────────────┘  |
|                            v                                       |
|  OUTPUT: Top-5 最相关且多样的文档 → 送入 LLM 生成回答               |
|                                                                    |
+------------------------------------------------------------------+
```

#### RRF（Reciprocal Rank Fusion）融合算法

```python
from typing import List, Dict
from collections import defaultdict

def reciprocal_rank_fusion(
    result_lists: List[List[str]], 
    k: int = 60
) -> List[tuple]:
    """
    RRF: Reciprocal Rank Fusion 多路召回融合
    
    算法原理:
        RRF_score(d) = Σ_{r in rankings} 1 / (k + rank_r(d))
    
    参数:
        result_lists: 各路召回的结果列表（已按相关性排序）
        k: 平滑常数，通常取 60
    
    返回:
        [(doc_id, fusion_score), ...] 按融合分数降序
    """
    scores = defaultdict(float)
    
    for ranking in result_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    
    # 按融合分数降序排列
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


# 使用示例
vector_results = ["doc_a", "doc_c", "doc_b", "doc_e"]   # 向量召回排序
bm25_results  = ["doc_b", "doc_a", "doc_d", "doc_f"]   # BM25 召回排序
keyword_results = ["doc_a", "doc_g", "doc_b"]          # 关键词召回排序

fused = reciprocal_rank_fusion([vector_results, bm25_results, keyword_results])
print("RRF 融合结果:", fused)
# doc_a: 1/(61+1) + 1/(61+2) + 1/(61+1) ≈ 0.0484 (第一)
# doc_b: 1/(61+3) + 1/(61+1) + 1/(61+3) ≈ 0.0474 (第二)
```

### 8.4.2 批量 Reranking 优化

```python
import torch
import numpy as np
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
import asyncio

class BatchReranker:
    """支持动态 Batching 的 Reranker"""
    
    def __init__(self, model, tokenizer, max_batch_size=64, max_seq_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        
        # 启用 Flash Attention 加速
        if hasattr(self.model.config, '_attn_implementation'):
            self.model.config._attn_implementation = "flash_attention_2"
        
    @torch.no_grad()
    def rerank_batch(self, query: str, documents: List[str], 
                     top_k: int = 10) -> List[Tuple[int, float, str]]:
        """
        批量 Rerank，自动处理超长文档截断
        """
        all_scores = []
        num_docs = len(documents)
        
        for batch_start in range(0, num_docs, self.max_batch_size):
            batch_end = min(batch_start + self.max_batch_size, num_docs)
            batch_docs = documents[batch_start:batch_end]
            
            # 构建输入对
            pairs = [[query, doc] for doc in batch_docs]
            inputs = self.tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                max_length=self.max_seq_length, 
                return_tensors="pt"
            ).to(self.model.device)
            
            # 推理
            scores = self.model(**inputs).logits.squeeze(-1)
            all_scores.extend(scores.cpu().tolist())
        
        # 索引排序
        scored = [(i, score, documents[i]) for i, score in enumerate(all_scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class StreamingReranker:
    """流式 Rerank：逐文档返回，适用于实时应用"""
    
    def __init__(self, reranker: BatchReranker):
        self.reranker = reranker
        
    async def rerank_stream(self, query: str, documents: List[str], 
                            top_k: int = 5):
        """
        异步流式 Rerank，每次返回一个已排序的结果
        适用于需要渐进式 UI 更新的场景
        """
        # 先快速打分
        scores = []
        for i, doc in enumerate(documents):
            score = await self._score_single(query, doc)
            scores.append((i, score, doc))
        
        # 实时排序输出
        scores.sort(key=lambda x: x[1], reverse=True)
        for rank, (idx, score, doc) in enumerate(scores[:top_k]):
            yield {"rank": rank + 1, "score": score, "document": doc, "index": idx}
            # 模拟渐进式返回
            await asyncio.sleep(0.01)
    
    async def _score_single(self, query: str, doc: str) -> float:
        """单文档异步打分"""
        result = self.reranker.rerank_batch(query, [doc], top_k=1)
        return result[0][1] if result else 0.0
```

### 8.4.3 多模型级联 Rerank

```
+------------------------------------------------------------------+
|                   多模型级联 Rerank 架构                            |
+------------------------------------------------------------------+
|                                                                    |
|  [Top-200 候选文档]                                                 |
|         |                                                          |
|         v                                                          |
|  +-------------------+                                             |
|  | Tier 1: 轻量 Reranker |  bge-reranker-base (278M)               |
|  | 延迟: ~150ms         |  筛选 Top-50                             |
|  | GPU: T4 / CPU 可运行  |                                         |
|  +-------------------+                                             |
|         |                                                          |
|         v  (Top-50)                                                |
|  +-------------------+                                             |
|  | Tier 2: 中量 Reranker |  bge-reranker-v2-m3 (568M)             |
|  | 延迟: ~200ms         |  筛选 Top-20                             |
|  | GPU: A10 / L4         |                                         |
|  +-------------------+                                             |
|         |                                                          |
|         v  (Top-20)                                                |
|  +-------------------+                                             |
|  | Tier 3: 重量 Reranker |  gte-qwen2-7b-instruct (7B)            |
|  | 延迟: ~300ms         |  筛选 Top-5                              |
|  | GPU: A100 / H100      |  (可选：极限精度场景)                    |
|  +-------------------+                                             |
|         |                                                          |
|         v                                                          |
|  [Top-5 最终结果] → 送入 LLM                                       |
|                                                                    |
|  总延迟: 150 + 200 + 300 = 650ms (满足 <1s 的交互体验要求)          |
|                                                                    |
+------------------------------------------------------------------+
```

### 8.4.4 性能优化策略

#### 模型量化

```python
from transformers import BitsAndBytesConfig
import torch

def load_quantized_reranker(model_name: str, quantization: str = "int8"):
    """
    加载量化后的 Reranker 模型
    
    量化选项:
    - fp32: 原始精度 (~4 bytes/param)
    - fp16: 半精度 (~2 bytes/param, 速度 2x)
    - int8: 8位量化 (~1 byte/param, 速度 3-4x, 精度损失 <1%)
    - int4: 4位量化 (~0.5 bytes/param, 速度 5-6x, 精度损失 1-3%)
    """
    if quantization == "int8":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True,
            torch_dtype=torch.float16
        )
    elif quantization == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
    
    return model
```

#### ONNX Runtime 导出与推理

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np

def export_to_onnx(model_name: str, output_path: str):
    """将 Reranker 导出为 ONNX 格式"""
    model = ORTModelForSequenceClassification.from_pretrained(
        model_name, 
        export=True
    )
    model.save_pretrained(output_path)
    print(f"ONNX 模型已保存至: {output_path}")


def infer_with_onnx(onnx_path: str, query: str, documents: List[str]):
    """使用 ONNX Runtime 推理"""
    model = ORTModelForSequenceClassification.from_pretrained(
        onnx_path,
        provider="CUDAExecutionProvider"  # 或 "CPUExecutionProvider"
    )
    tokenizer = AutoTokenizer.from_pretrained(onnx_path)
    
    pairs = [[query, doc] for doc in documents]
    inputs = tokenizer(pairs, padding=True, truncation=True, 
                       max_length=512, return_tensors="pt")
    
    # ONNX 推理通常比 PyTorch 原生推理快 1.5-2x
    outputs = model(**inputs)
    scores = outputs.logits.squeeze(-1)
    return scores.cpu().numpy()
```

#### Caching 策略

```python
from functools import lru_cache
import hashlib
import json

class CachedReranker:
    """带缓存的 Reranker，减少重复计算"""
    
    def __init__(self, reranker, cache_size: int = 10000):
        self.reranker = reranker
        self.cache = {}  # 简单的 Dict Cache
        
    @staticmethod
    def _cache_key(query: str, docs: List[str]) -> str:
        """生成缓存键"""
        content = json.dumps({"q": query, "d": docs}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def rerank(self, query: str, documents: List[str], 
               top_k: int = 10, use_cache: bool = True):
        """带缓存的 Rerank"""
        cache_key = self._cache_key(query, documents)
        
        if use_cache and cache_key in self.cache:
            print(f"[Cache Hit] 返回缓存结果")
            return self.cache[cache_key][:top_k]
        
        results = self.reranker.rerank(query, documents, top_k=top_k)
        
        if len(self.cache) < 10000:  # 防止 OOM
            self.cache[cache_key] = results
            
        return results
```

---

## 8.5 高级 Rerank 技术

### 8.5.1 Pointwise / Pairwise / Listwise 三种排序范式

```
+------------------------------------------------------------------+
|               三种排序范式对比                                      |
+------------------------------------------------------------------+
|                                                                    |
|  ┌──────────────────────────────────────────────────────────────┐ |
|  │ Pointwise（逐点）                                             │ |
|  │ 输入: (Query, Doc_i) → 模型 → 相关性得分 S_i                   │ |
|  │ 损失: 回归损失 (MSE) 或二分类损失 (BCE)                        │ |
|  │ 优点: 简单直接，推理速度快                                      │ |
|  │ 缺点: 不考虑文档间的相对关系，易产生绝对分数偏差                   │ |
|  │ 代表: BGE Reranker, Jina Reranker                             │ |
|  └──────────────────────────────────────────────────────────────┘ |
|                                                                    |
|  ┌──────────────────────────────────────────────────────────────┐ |
|  │ Pairwise（成对）                                               │ |
|  │ 输入: (Query, Doc_i, Doc_j) → 模型 → P(Doc_i > Doc_j | Query)  │ |
|  │ 损失: RankNet / LambdaRank                                    │ |
|  │ 优点: 学习文档间的相对顺序，更贴近排序目标                        │ |
|  │ 缺点: 训练复杂度 O(n^2)，需要构造文档对                          │ |
|  │ 代表: DuoT5, monoBERT (部分)                                   │ |
|  └──────────────────────────────────────────────────────────────┘ |
|                                                                    |
|  ┌──────────────────────────────────────────────────────────────┐ |
|  │ Listwise（列表）                                               │ |
|  │ 输入: (Query, [Doc_1, Doc_2, ..., Doc_n]) → 模型 → 排列概率分布  │ |
|  │ 损失: ListMLE / ListNet / Softmax Cross-Entropy               │ |
|  │ 优点: 全局视角，直接优化整个排序列表                              │ |
|  │ 缺点: 输入长度受限（Transformer 复杂度 O(L^2)），长列表开销大      │ |
|  │ 代表: RankGPT, RankVicuna, LiT5                               │ |
|  └──────────────────────────────────────────────────────────────┘ |
|                                                                    |
+------------------------------------------------------------------+
```

```
+------------------------------------------------------------------+
|          Pointwise vs Pairwise vs Listwise 实战对比表              |
+------------------------------------------------------------------+
|                                                                    |
|  维度            Pointwise       Pairwise        Listwise         |
|  ───────────────────────────────────────────────────────────────  |
|  训练复杂度      低             中              高               |
|  推理复杂度      低             中              高               |
|  NDCG@10 提升   基准            +1~3%           +3~5%            |
|  GPU 显存需求    低             中              高               |
|  适合文档数      任意           数十            十以内            |
|  中文生态支持    丰富           较少            极少(LLM方案)     |
|  工业界使用率    主流(>80%)     小范围(<15%)    实验阶段(<5%)    |
|                                                                    |
+------------------------------------------------------------------+
```

### 8.5.2 LLM-as-Reranker

LLM（大语言模型）正在成为 Rerank 的一种新范式。通过精心设计的 Prompt，GPT-4、Claude 等模型可以直接评估 Query-Document 相关性。

#### RankGPT 方法

```python
from openai import OpenAI
from typing import List, Tuple
import json

class RankGPT:
    """使用 GPT-4 作为 Reranker（基于 RankGPT 论文）"""
    
    SYSTEM_PROMPT = """你是一个专业的文档排序助手。
你的任务是根据用户查询的相关性，对给定的文档列表进行排序。

规则：
1. 仔细阅读用户查询和所有文档
2. 按相关性从高到低排列文档
3. 只返回文档编号列表（用逗号分隔），不要输出其他内容
4. 如果文档完全不相关，将其排在最后"""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def rerank(self, query: str, documents: List[str], 
               top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        使用滑动窗口策略处理超过 LLM 上下文窗口的文档列表
        """
        # 第1步：编号文档
        numbered_docs = "\n".join([
            f"[{i+1}] {doc[:500]}"  # 截断长文档
            for i, doc in enumerate(documents)
        ])
        
        # 第2步：调用 LLM
        user_prompt = f"用户查询: {query}\n\n候选文档:\n{numbered_docs}\n\n请按相关性排序，只返回文档编号。"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        # 第3步：解析排序结果
        rank_text = response.choices[0].message.content.strip()
        try:
            ranks = [int(x.strip()) for x in rank_text.replace("[","").replace("]","").split(",")]
        except ValueError:
            ranks = list(range(1, len(documents) + 1))
        
        # 第4步：构造返回结果（模拟分数：位置越前分数越高）
        scores = []
        for rank, doc_idx in enumerate(ranks[:top_k]):
            normalized_score = 1.0 - (rank / len(ranks))
            scores.append((doc_idx - 1, normalized_score, documents[doc_idx - 1]))
            
        return scores


# 使用示例
rankgpt = RankGPT(api_key="sk-xxx")
results = rankgpt.rerank(
    "Redis 集群最少需要几个节点？",
    candidates[:10],   # 一般给 LLM 的候选不超过 20 个
    top_k=5
)
```

#### LLM Reranker 的优缺点

| 优点 | 缺点 |
|------|------|
| 零样本能力极强，无需训练 | 延迟高（500ms-2s/请求） |
| 理解复杂语义和隐式约束 | 成本高（GPT-4 ~$0.01/次） |
| 可处理多语言混合查询 | 输出不稳定（需结构化解析） |
| 可提供排序理由（Explainable Rerank） | 上下文窗口限制候选文档数 |
| 适合冷启动和快速原型 | 不适合高并发在线服务 |

### 8.5.3 MMR（Maximal Marginal Relevance）多样性优化

Rerank 的纯相关性排序可能导致 Top-K 结果高度同质（例如返回 5 篇关于"Redis集群节点数"的文档，忽略了"Redis集群高可用方案"这种互补信息）。MMR 算法在相关性和多样性之间寻求平衡。

```
MMR 核心公式：

MMR(d_i) = λ · Relevance(d_i, Q) - (1-λ) · max_{d_j ∈ S} Similarity(d_i, d_j)

参数:
  d_i      : 候选文档
  Q        : 用户查询
  S        : 已选文档集合
  λ        : 相关性-多样性权衡系数 (0~1)
             λ=1.0 → 纯相关性排序
             λ=0.0 → 纯多样性排序
             λ=0.7 → 推荐的企业默认值
```

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class MMRReranker:
    """MMR 多样性重排序"""
    
    def __init__(self, embedder_model: str = "BAAI/bge-large-zh-v1.5", 
                 lambda_param: float = 0.7):
        self.embedder = SentenceTransformer(embedder_model)
        self.lambda_param = lambda_param
        
    def rerank_with_diversity(
        self, 
        query: str, 
        documents: List[str], 
        relevance_scores: List[float],
        top_k: int = 5
    ) -> List[Tuple[int, float, str, float]]:
        """
        MMR 多样性重排序
        
        参数:
            query: 用户查询
            documents: 候选文档列表
            relevance_scores: 初排分数（来自 Cross Encoder）
            top_k: 返回文档数
            
        返回:
            [(index, mmr_score, document, relevance_score), ...]
        """
        # 步骤1: 计算所有文档的嵌入向量（用于文档间相似度）
        doc_embeddings = self.embedder.encode(documents, normalize_embeddings=True)
        
        # 步骤2: 归一化相关性分数到 [0, 1]
        rel_scores = np.array(relevance_scores)
        rel_scores = (rel_scores - rel_scores.min()) / (rel_scores.max() - rel_scores.min() + 1e-8)
        
        # 步骤3: 迭代选择
        remaining = list(range(len(documents)))
        selected = []
        
        for _ in range(min(top_k, len(documents))):
            mmr_scores = []
            
            for idx in remaining:
                relevance = rel_scores[idx]
                
                if not selected:
                    # 第一个文档：只看相关性
                    diversity_penalty = 0
                else:
                    # 已选文档的最大相似度（惩罚项）
                    similarities = [
                        cosine_similarity(
                            [doc_embeddings[idx]], 
                            [doc_embeddings[sel]]
                        )[0][0]
                        for sel in selected
                    ]
                    diversity_penalty = max(similarities)
                
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * diversity_penalty
                mmr_scores.append(mmr)
            
            # 选择 MMR 分数最高的文档
            best_local_idx = np.argmax(mmr_scores)
            best_global_idx = remaining[best_local_idx]
            
            selected.append(best_global_idx)
            remaining.remove(best_global_idx)
        
        # 步骤4: 返回结果
        results = []
        for rank, idx in enumerate(selected):
            results.append((
                idx, 
                relevance_scores[idx],  # 原始相关性分数
                documents[idx],
                rel_scores[idx]         # 归一化相关性分数
            ))
        
        return results


# 使用示例
mmr = MMRReranker(lambda_param=0.7)
documents = [
    "Redis 集群最少需要 6 个节点，3 主 3 从...",
    "Redis 集群的节点配置至少需要 6 个实例...",
    "Redis 集群高可用架构设计方案...",
    "Redis 集群最小节点数计算方式...",
    "Redis 哨兵模式与集群模式对比...",
]
scores = [0.95, 0.93, 0.70, 0.92, 0.68]

results = mmr.rerank_with_diversity(
    "Redis 集群最少需要几个节点？", documents, scores, top_k=3
)

for rank, (idx, score, doc, _) in enumerate(results, 1):
    print(f"Rank {rank} [Score:{score:.3f}] {doc[:60]}...")
```

### 8.5.4 多维度 Rerank

在实际企业场景中，仅靠语义相关性不够，需要综合考虑多维度信号：

```
+------------------------------------------------------------------+
|                 多维度 Rerank 信号融合架构                          |
+------------------------------------------------------------------+
|                                                                    |
|  最终得分 = W_sem · Score_sem               # 语义相关性           |
|           + W_time · Score_time             # 时效性               |
|           + W_auth · Score_authority          # 权威性               |
|           + W_diverse · Score_diversity      # 多样性               |
|           + W_qual · Score_quality           # 文档质量             |
|           + W_biz  · Score_business           # 业务权重             |
|                                                                    |
|  ┌─────────────────────────────────────────────────────────────┐  |
|  │ 信号          来源                  归一化方式                │  |
|  │─────────────────────────────────────────────────────────────│  |
|  │ 语义相关性     Cross Encoder 输出    Sigmoid → [0, 1]       │  |
|  │ 时效性         doc_publish_date      指数衰减 e^{-λ·Δt}      │  |
|  │ 权威性         source_rank (预设)    MinMax 归一化            │  |
|  │ 多样性         MMR 分数              公式计算                 │  |
|  │ 文档质量       readability_score     MinMax 归一化            │  |
|  │ 业务权重       biz_config (规则)     [0.5, 1.5] 乘子         │  |
|  └─────────────────────────────────────────────────────────────┘  |
|                                                                    |
+------------------------------------------------------------------+
```

```python
from datetime import datetime
from typing import Dict, Any

class MultiDimensionReranker:
    """多维度融合 Reranker"""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "semantic": 0.40,
            "freshness": 0.15,
            "authority": 0.15,
            "diversity": 0.15,
            "quality": 0.10,
            "business": 0.05
        }
        
    def compute_final_score(self, doc_id: str, signals: Dict[str, float]) -> float:
        """计算多维融合得分"""
        score = 0.0
        for dim, weight in self.weights.items():
            if dim in signals:
                score += weight * signals[dim]
        return score
    
    def freshness_score(self, publish_date: datetime, 
                        half_life_days: int = 365) -> float:
        """时效性分数：指数衰减"""
        days_elapsed = (datetime.now() - publish_date).days
        return np.exp(-np.log(2) * max(days_elapsed, 0) / half_life_days)
    
    def rerank(self, query: str, candidates: List[Dict[str, Any]], 
               top_k: int = 5) -> List[Dict[str, Any]]:
        """多维度综合排序"""
        for doc in candidates:
            doc["_final_score"] = self.compute_final_score(doc["id"], doc["signals"])
        
        candidates.sort(key=lambda x: x["_final_score"], reverse=True)
        return candidates[:top_k]
```

---

## 8.6 企业级部署

### 8.6.1 Rerank 服务架构

```
+------------------------------------------------------------------+
|                  企业级 Rerank 微服务架构                           |
+------------------------------------------------------------------+
|                                                                    |
|                        [API Gateway]                               |
|                             |                                      |
|                +------------+-----------+                          |
|                |                        |                          |
|                v                        v                          |
|        [Rerank Service]          [Monitoring Stack]               |
|        (FastAPI/gRPC)            (Prometheus + Grafana)           |
|                |                        |                          |
|        +-------+-------+         [AlertManager]                    |
|        |               |              |                            |
|        v               v         [Slack/飞书告警]                  |
|  [Model Pool]    [Cache Layer]                                     |
|        |               |                                           |
|  +-----+-----+    +----+----+                                     |
|  | GPU Node 1|    | Redis   |                                     |
|  | GPU Node 2|    | Cluster |                                     |
|  | GPU Node N|    +---------+                                     |
|  +-----------+                                                     |
|        |                                                           |
|        v                                                           |
|  [Model Registry]                                                  |
|  (bge-reranker-v2-m3, jina-reranker-v3, gte-qwen2-7b)              |
|                                                                    |
+------------------------------------------------------------------+
```

### 8.6.2 生产级 Rerank 服务实现

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio
import time
from contextlib import asynccontextmanager
import torch

# ------ 数据模型 ------
class RerankRequest(BaseModel):
    query: str = Field(..., description="用户查询")
    documents: List[str] = Field(..., min_items=1, max_items=500, description="候选文档列表")
    top_k: int = Field(default=5, ge=1, le=50, description="返回结果数")
    model: str = Field(default="bge-reranker-v2-m3", description="模型名称")
    return_documents: bool = Field(default=True, description="是否返回文档原文")
    lambda_mmr: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="MMR 多样性系数")

class RerankResult(BaseModel):
    index: int
    score: float
    document: Optional[str] = None

class RerankResponse(BaseModel):
    results: List[RerankResult]
    model: str
    latency_ms: float
    request_id: str

# ------ 模型管理器 ------
class ModelManager:
    """多模型生命周期管理"""
    
    def __init__(self):
        self.models = {}
        self.model_lock = asyncio.Lock()
    
    async def get_model(self, model_name: str):
        """模型懒加载 + 并发安全"""
        if model_name not in self.models:
            async with self.model_lock:
                if model_name not in self.models:  # Double-check
                    self.models[model_name] = self._load_model(model_name)
        return self.models[model_name]
    
    def _load_model(self, model_name: str):
        """加载模型到 GPU"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        return {"tokenizer": tokenizer, "model": model, "device": device}

model_manager = ModelManager()

# ------ FastAPI 应用 ------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时预热模型"""
    print("预热 Rerank 模型...")
    await model_manager.get_model("BAAI/bge-reranker-v2-m3")
    print("模型预热完成，服务就绪")
    yield

app = FastAPI(title="Enterprise Rerank Service", version="2.0.0", lifespan=lifespan)

@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank_endpoint(request: RerankRequest):
    """Rerank API 端点"""
    start_time = time.time()
    
    try:
        # 获取模型
        model_resource = await model_manager.get_model(request.model)
        tokenizer = model_resource["tokenizer"]
        model = model_resource["model"]
        
        # 批量打分
        with torch.no_grad():
            scores = []
            batch_size = 32
            for i in range(0, len(request.documents), batch_size):
                batch = [[request.query, doc] for doc in 
                         request.documents[i:i+batch_size]]
                inputs = tokenizer(batch, padding=True, truncation=True, 
                                   max_length=8192, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                batch_scores = model(**inputs).logits.squeeze(-1)
                scores.extend(batch_scores.cpu().tolist())
        
        # 排序
        scored = [(i, s) for i, s in enumerate(scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # MMR 多样性优化（可选）
        if request.lambda_mmr is not None:
            scored = apply_mmr_diversity(
                scored, request.documents, request.lambda_mmr
            )
        
        # 构建响应
        results = []
        for rank, (idx, score) in enumerate(scored[:request.top_k]):
            results.append(RerankResult(
                index=idx,
                score=round(score, 6),
                document=request.documents[idx] if request.return_documents else None
            ))
        
        latency = (time.time() - start_time) * 1000
        
        return RerankResponse(
            results=results,
            model=request.model,
            latency_ms=round(latency, 2),
            request_id=f"req_{int(start_time)}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "models_loaded": list(model_manager.models.keys()),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
```

### 8.6.3 负载均衡策略

```
+------------------------------------------------------------------+
|                  Rerank 负载均衡架构                                |
+------------------------------------------------------------------+
|                                                                    |
|  负载均衡策略: Least Connection（最小连接数）                        |
|                                                                    |
|  [Nginx / Envoy / HAProxy]                                        |
|          |                                                         |
|          +----+-----+-----+-----+                                  |
|               |     |     |     |                                  |
|               v     v     v     v                                  |
|          [Node1] [Node2] [Node3] [Node4]   ← GPU 推理节点          |
|           A10    A10    A100   A100                                |
|                                                                    |
|  健康检查: 每 10s GET /health                                       |
|  故障转移: 连续 3 次失败 → 摘除节点                                 |
|  会话亲和: IP Hash → 同一用户的路由到同一节点(利用缓存)              |
|                                                                    |
+------------------------------------------------------------------+
```

#### Nginx 配置示例

```nginx
upstream rerank_backend {
    least_conn;  # 最小连接数策略
    server 10.0.1.101:8000 weight=1 max_fails=3 fail_timeout=30s;
    server 10.0.1.102:8000 weight=1 max_fails=3 fail_timeout=30s;
    server 10.0.1.103:8000 weight=2 max_fails=3 fail_timeout=30s;  # A100 节点,权重更高
    server 10.0.1.104:8000 weight=2 max_fails=3 fail_timeout=30s;
    
    keepalive 32;  # 连接池
}

server {
    listen 80;
    server_name rerank.internal.example.com;
    
    location /v1/rerank {
        proxy_pass http://rerank_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 30s;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}
```

### 8.6.4 延迟预算分析

```
+------------------------------------------------------------------+
|                  端到端 RAG 延迟预算分配                            |
+------------------------------------------------------------------+
|                                                                    |
|  目标总延迟: < 2.0 秒 (用户可接受的交互等待)                        |
|                                                                    |
|  ┌──────────────────────────────────────────────────────────────┐ |
|  │ 阶段                     延迟预算    占比    备注              │ |
|  │──────────────────────────────────────────────────────────────│ |
|  │ 1. Query 理解/重写        < 100ms     5%    规则/小模型        │ |
|  │ 2. 向量化 (Embedding)     < 50ms      2.5%  API 调用          │ |
|  │ 3. 多路召回 (向量+BM25)   < 200ms     10%   并行执行           │ |
|  │ 4. RRF 融合 + 去重        < 20ms      1%    CPU 轻量计算      │ |
|  │ 5. Rerank 精排            < 500ms     25%   GPU 推理           │ |
|  │ 6. 上下文构建             < 30ms      1.5%  模板拼接           │ |
|  │ 7. LLM 推理               < 1000ms    50%   主流 LLM 推理时间  │ |
|  │ 8. 后处理/安全过滤         < 100ms     5%    规则/小模型        │ |
|  │──────────────────────────────────────────────────────────────│ |
|  │ 总计                      < 2000ms    100%                    │ |
|  └──────────────────────────────────────────────────────────────┘ |
|                                                                    |
|  Rerank 优化空间:                                                   |
|  - batch_size 从 1 → 32: 吞吐提升 8x                               |
|  - FP16 量化: 延迟降低 40%                                         |
|  - ONNX Runtime: 延迟降低 30%                                      |
|  - Flash Attention: 长序列延迟降低 50%                              |
|  - 级联策略: 200→50→10 的 3 级筛选可节省 200ms                     |
|                                                                    |
+------------------------------------------------------------------+
```

### 8.6.5 监控与质量指标

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response

# ------ Prometheus 指标定义 ------

# 请求计数
rerank_requests_total = Counter(
    "rerank_requests_total", 
    "Total rerank requests",
    ["model", "status"]  # status: success / error
)

# 延迟直方图
rerank_latency_seconds = Histogram(
    "rerank_latency_seconds",
    "Rerank request latency in seconds",
    ["model"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

# 批次大小
rerank_batch_size = Histogram(
    "rerank_batch_size",
    "Number of documents per rerank request",
    buckets=[1, 5, 10, 20, 50, 100, 200, 500]
)

# GPU 利用率
gpu_utilization = Gauge(
    "gpu_utilization_percent",
    "GPU utilization percentage",
    ["gpu_id"]
)

# Rerank 质量指标（离线）
rerank_ndcg = Gauge(
    "rerank_ndcg_at_10",
    "NDCG@10 score for rerank quality (offline evaluation)",
    ["model"]
)


@app.get("/metrics")
async def metrics():
    """Prometheus 指标暴露端点"""
    # 更新 GPU 指标
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            util = torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0
            gpu_utilization.labels(gpu_id=str(i)).set(util)
    
    return Response(content=generate_latest(), media_type="text/plain")
```

#### Rerank 质量监控仪表盘（Grafana Panel 设计）

```
+------------------------------------------------------------------+
|                 Rerank 质量监控仪表盘                               |
+------------------------------------------------------------------+
|                                                                    |
|  ┌─────────────────────┐  ┌─────────────────────┐                 |
|  │ P50/P95/P99 延迟     │  │ QPS (每秒请求数)      │                |
|  │ [折线图 - 时间序列]   │  │ [柱状图 - 时间序列]   │                |
|  └─────────────────────┘  └─────────────────────┘                 |
|                                                                    |
|  ┌─────────────────────┐  ┌─────────────────────┐                 |
|  │ 请求成功率(%)         │  │ GPU 利用率 (%)       │                |
|  │ [仪表盘 - 实时]       │  │ [面积图 - 时间序列]   │                |
|  └─────────────────────┘  └─────────────────────┘                 |
|                                                                    |
|  ┌──────────────────────────────────────────────┐                |
|  │ 候选文档数分布 (直方图)                         │               |
|  │ [Histogram - batch_size]                      │               |
|  └──────────────────────────────────────────────┘                |
|                                                                    |
|  ┌──────────────────────────────────────────────┐                |
|  │ 离线质量指标 (周级)                             │               |
|  │ ──────────────────────────────────────────── │               |
|  │ Model           NDCG@10  MRR@10  Recall@10   │               |
|  │ bge-reranker-v2-m3  52.7   68.3     85.2      │               |
|  │ gte-qwen2-7b        56.8   72.1     88.5      │               |
|  │ cohere-v3.5         55.9   70.8     87.1      │               |
|  └──────────────────────────────────────────────┘                |
|                                                                    |
+------------------------------------------------------------------+
```

---

## 8.7 企业最佳实践

### 8.7.1 架构实践清单

```
+------------------------------------------------------------------+
|                   Rerank 企业最佳实践清单                            |
+------------------------------------------------------------------+
|                                                                    |
|  [选型]                                                             |
|  □ 中文场景首选 bge-reranker-v2-m3                                 |
|  □ 多语言场景备选 jina-reranker-v3 或 Cohere API                   |
|  □ 极限精度场景使用 gte-qwen2-7b + 级联                             |
|  □ 低资源场景使用 ONNX 量化版 bge-reranker-base                    |
|                                                                    |
|  [流水线]                                                           |
|  □ 召回量 K 控制在 100-200，避免 Rerank 过载                        |
|  □ 使用 RRF 融合多种召回源                                          |
|  □ 实现多级级联：轻量 Reranker → 重量 Reranker                     |
|  □ 对热点 Query 开启 Rerank 结果缓存                                |
|                                                                    |
|  [性能]                                                             |
|  □ 生产环境务必使用 FP16 或 INT8 量化                               |
|  □ 开启 Flash Attention 2 加速长文本推理                            |
|  □ 使用 Dynamic Batching 提升 GPU 利用率                            |
|  □ Rerank 服务与 LLM 推理服务共享 GPU 时使用 MPS/MIG 隔离           |
|                                                                    |
|  [质量]                                                             |
|  □ 建立标注数据集，每周离线评测 NDCG@10                             |
|  □ 监控 Rerank 分数分布，异常漂移应告警                              |
|  □ 定期进行 A/B 测试（新模型 vs 旧模型）                             |
|  □ 收集用户反馈（点赞/点踩）作为在线质量信号                         |
|                                                                    |
|  [运维]                                                             |
|  □ Rerank 服务独立部署，与检索和生成解耦                            |
|  □ 配置健康检查 + 自动重启                                          |
|  □ 设置延迟 SLO: P95 < 500ms                                       |
|  □ 使用 HPA (Horizontal Pod Autoscaler) 根据 QPS 自动扩缩容         |
|                                                                    |
+------------------------------------------------------------------+
```

### 8.7.2 常见故障排查

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| Rerank 延迟突增 | 候选文档数暴增 | 限制最大输入文档数为 200 |
| Rerank 得分异常 | 模型需要对输入做 norm | 确认 tokenizer 的 padding/truncation 配置 |
| GPU OOM | 文档过长 | 对超长文档做滑动窗口分块，取 max score |
| 分数全接近 0.5 | 模型输出未经 sigmoid | 检查 `model.config.problem_type` |
| 中文排序不准 | 使用了纯英文 Reranker | 切换到 bge-reranker-v2-m3 或 jina-v3 |

---

## 8.8 面试高频问题

**Q1: 为什么不能只靠 Embedding 召回？Rerank 解决了什么核心问题？**

A: Embedding 召回使用 Bi-Encoder 架构，Query 和 Document 独立编码为固定维度向量，通过点积/余弦计算相似度。这个过程存在"信息瓶颈"——Query 和 Doc 之间从未发生 Token 级别的深度交互，无法区分细粒度的语义差异（如否定、词序、精确约束）。Rerank 使用 Cross Encoder 架构，将 Query-Doc 拼接后送入 Transformer，通过 Self-Attention 实现每个 Token 的全对全交互，弥合了"语义相似"与"真实相关"之间的鸿沟。

**Q2: ColBERT 的 MaxSim 操作是如何工作的？为什么它是 Bi-Encoder 和 Cross Encoder 的折中方案？**

A: MaxSim 的核心思想是延迟交互（Late Interaction）。Query 的每个 Token 向量与 Document 的所有 Token 向量计算点积，取最大值（即找到最匹配的 Doc Token），然后将所有 Query Token 的 MaxSim 分数求和。它在**不影响离线索引**（Doc Token 向量可预计算）的前提下实现了 Token 级匹配，精度介于 Bi-Encoder（无 Token 交互）和 Cross Encoder（全对全交互）之间。

**Q3: 如何选择 Rerank 模型？**

A: 遵循以下决策路径：(1) 评估语言需求（仅中文/多语言）；(2) 评估 GPU 资源（T4 以下选轻量模型，A10 可选标准模型，A100 可选重型模型）；(3) 评估精度要求（通用场景 bge-reranker-v2-m3 足够，金融/法律等高精度场景可选 gte-qwen2-7b）；(4) 评估是否接受 API 调用（可接受则 Cohere Rerank 免运维）。

**Q4: 如何优化 Rerank 的推理延迟？**

A: 核心策略包括：(1) FP16/INT8 量化降低计算量；(2) ONNX Runtime 或 TensorRT 推理加速；(3) Flash Attention 2 加速长文本 Attention；(4) Dynamic Batching 提升 GPU 利用率；(5) 级联策略——先用轻量模型筛选再交给重量模型；(6) 对热点 Query 缓存 Rerank 结果。

**Q5: MMR 算法的作用是什么？Lambda 参数如何设置？**

A: MMR 在保持语义相关性的前提下引入多样性惩罚，避免 Top-K 结果内容同质。Lambda=1.0 退化为纯相关性排序；Lambda=0.0 为纯多样性（不推荐）。企业场景推荐 Lambda=0.6~0.8。过低的 Lambda 会引入不相关但"不一样"的文档，损害 RAG 答案质量。

---

## 8.9 总结

Rerank 是 RAG 系统中连接"粗召回"与"精生成"的关键桥梁。它的核心价值在于**用更强的模型（Cross Encoder 的 Token 级交互）在更小的候选集上（Top 100-200）做更精确的判断（Pointwise 相关性打分）**。

**核心要点回顾**：

1. **语义鸿沟是真实的**：Embedding 相似度不等于答案相关性，Rerank 是弥合这一鸿沟的工业标准方案
2. **Cross Encoder 是当前 Rerank 的主流架构**：全对全 Token 交互带来显著精度提升
3. **ColBERT 是有价值的折中**：适合对离线索引速度有高要求、但需要 Token 级匹配的场景
4. **模型选择应基于实际需求**：中文优先 bge-reranker-v2-m3，多语言优先 jina-v3，极限精度优先 gte-qwen2-7b
5. **多级级联是最优实践**：轻量模型快速筛选 → 重量模型精细排序，平衡精度与成本
6. **企业部署需要全链路考量**：从量化加速、负载均衡、延迟预算到质量监控，缺一不可
7. **LLM-as-Reranker 是未来趋势**：但当前延迟和成本限制了其大规模在线应用，适合离线评估和小批量高价值场景

Rerank 技术正处于快速发展期，从传统的 Cross Encoder 到新兴的 LLM-as-Reranker、多模态 Reranker、个性化 Reranker，未来将朝着更高精度、更低延迟、更强可解释性的方向演进。

---

> **下一章预告**：第九章《知识库构建与索引系统》——深入探讨文档解析、切片策略、元数据管理、向量索引（HNSW/IVF/DiskANN）、动态更新、增量索引等核心技术。
