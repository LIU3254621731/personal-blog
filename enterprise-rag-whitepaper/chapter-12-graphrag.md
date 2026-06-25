# 第12章 GraphRAG：知识图谱增强的检索生成

> **摘要**：传统RAG系统依赖向量相似度进行检索，在面对多跳推理、实体关系问答、全局摘要等复杂任务时存在根本性局限。GraphRAG将知识图谱的结构化语义理解能力引入RAG管道，通过图遍历、社区检测、混合检索等技术，实现了对文本证据之间深层关联的显式建模。本章将从基础原理、图谱构建、图数据库选型、检索方法、工程实践等维度，系统阐述GraphRAG的完整技术体系。

---

## 12.1 GraphRAG基础原理

### 12.1.1 什么是GraphRAG

GraphRAG（Graph-based Retrieval-Augmented Generation）是将**知识图谱**作为RAG管道的核心检索索引，利用图结构捕获实体、关系和文档之间的语义拓扑，以提升检索精度和推理深度的技术范式。

其核心思想源自微软研究院2024年发表的论文《From Local to Global: A Graph RAG Approach to Query-Focused Summarization》。论文指出，传统RAG仅检索与查询最相似的Top-K文本块，但这种方式在处理需要跨多个文档片段进行综合推理的查询时，会丢失全局上下文结构。GraphRAG通过在文档集上构建知识图谱，并执行**社区检测**和**社区摘要**，使LLM能够基于图谱社区而非孤立文本块来生成答案。

从架构角度，GraphRAG = 知识图谱构建层 + 图检索引擎 + LLM推理层：

```
+------------------------------------------------------------------+
|                        GraphRAG 系统架构                            |
+------------------------------------------------------------------+
|                                                                    |
|   [原始文档集]                                                       |
|        |                                                           |
|        v                                                           |
|   +-----------+    +-----------+    +--------------+               |
|   | 实体抽取   |--->| 关系抽取   |--->| 实体消歧/链接 |              |
|   +-----------+    +-----------+    +--------------+               |
|                                           |                        |
|                                           v                        |
|   +----------------------------------------------------------------+ |
|   |                    知识图谱 (Graph DB)                          | |
|   |  +---------+     +---------+     +-----------+                | |
|   |  | Node    |<--->| Edge    |<--->| Property  |                | |
|   |  | (Entity) |    | (Relation)|   | (Attribute)|               | |
|   |  +---------+     +---------+     +-----------+                | |
|   +----------------------------------------------------------------+ |
|                      |                     |                        |
|                      v                     v                        |
|   +---------------------------+   +---------------------------+     |
|   |  图检索引擎                |   |  向量检索引擎 (混合模式)    |     |
|   |  - BFS/DFS 遍历           |   |  - 嵌入相似度检索           |     |
|   |  - 社区检测 (Leiden)      |   |  - 混合排序融合             |     |
|   |  - 最短路径/多跳推理      |   |                             |     |
|   +---------------------------+   +---------------------------+     |
|                      |                     |                        |
|                      v                     v                        |
|   +----------------------------------------------------------------+ |
|   |                   检索结果融合 & 重排序                          | |
|   +----------------------------------------------------------------+ |
|                                    |                                |
|                                    v                                |
|   +----------------------------------------------------------------+ |
|   |               LLM 推理层 (Answer Generation)                    | |
|   +----------------------------------------------------------------+ |
|                                                                    |
+------------------------------------------------------------------+
```

### 12.1.2 为什么需要GraphRAG

传统基于向量的RAG（VectorRAG）尽管在绝大多数简单问答场景中表现优异，但在以下场景中存在根本性局限：

| 场景 | VectorRAG的局限 | GraphRAG的优势 |
|------|----------------|---------------|
| **多跳推理** | 单次向量检索只能找回与查询语义相似的文本块，无法跟踪"A->B->C"的推理链 | 通过图遍历执行显式的多跳路径搜索，精确跟踪实体间的关系链 |
| **实体关系问答** | 无法区分"Alice投资了Bob的公司"与"Bob投资了Alice的公司" | 关系边包含方向性，可以精确表达实体间的二元关系 |
| **全局摘要/趋势分析** | 缺乏对整个文档集的宏观理解，仅返回局部相似的片段 | 社区检测提供从局部到全局的分层摘要能力 |
| **复杂约束查询** | 无法处理如"找出与张三合作过且同在李四董事会任职的人"这类结构化约束 | 图查询语言（Cypher/SPARQL）天然支持复杂约束的图模式匹配 |
| **低频实体/长尾知识** | 向量检索对罕见实体容易失效，因为嵌入模型的训练数据中缺少对应表示 | 图结构基于精确匹配，不受实体频率影响 |
| **可解释性** | 仅给出相似度分数，难以解释检索逻辑 | 图路径即为推理路径，天然具备可解释性 |

### 12.1.3 知识图谱的定义与分类

知识图谱（Knowledge Graph）是一种用图结构表示知识的语义网络，其中**节点**代表实体或概念，**边**代表实体间的关系，**属性**存储实体/关系的附加信息。

在企业RAG场景中，知识图谱主要分为三类：

```
+------------------------------------------------------------------+
|                    企业知识图谱分类                                  |
+------------------------------------------------------------------+
|                                                                    |
|  1. 实体图谱 (Entity Graph)                                        |
|     +----------+      works_at      +----------+                  |
|     | 张三     | -----------------> | ABC公司  |                  |
|     +----------+                    +----------+                  |
|          |                               |                        |
|          | colleague_of                  | located_in             |
|          v                               v                        |
|     +----------+                    +----------+                  |
|     | 李四     |                    | 北京     |                  |
|     +----------+                    +----------+                  |
|                                                                    |
|  2. 文档图谱 (Document Graph)                                      |
|     +----------+      cites         +----------+                  |
|     | 文档A    | -----------------> | 文档B    |                  |
|     +----------+                    +----------+                  |
|          |                               |                        |
|          | contains_entity               | contains_entity        |
|          v                               v                        |
|     +----------+                    +----------+                  |
|     | 实体X    | <--- co_occurs --- | 实体Y    |                  |
|     +----------+                    +----------+                  |
|                                                                    |
|  3. 概念图谱 (Concept Graph)                                       |
|     +----------+     broader_than    +----------+                  |
|     | 深度学习  | <------------------ | 机器学习  |                 |
|     +----------+                    +----------+                  |
|          |                               |                        |
|          | related_to                    | related_to             |
|          v                               v                        |
|     +----------+                    +----------+                  |
|     | Transformer|                  | 神经网络  |                  |
|     +----------+                    +----------+                  |
|                                                                    |
+------------------------------------------------------------------+
```

### 12.1.4 GraphRAG vs 传统RAG：架构与能力对比

| 维度 | 传统VectorRAG | GraphRAG | 混合RAG |
|------|-------------|----------|---------|
| **索引结构** | 向量索引 (HNSW/IVF) | 图数据库 + 向量索引 | 双索引融合 |
| **检索基元** | 语义相似度 (余弦) | 图遍历 + 图模式匹配 + 语义相似度 | 图路径 + 语义排序 |
| **推理深度** | 单跳（查询→文本块） | 多跳（查询→实体→关系→实体→…） | 多跳 + 语义消歧 |
| **查询类型** | 自然语言模糊查询 | 结构化图查询 (Cypher/SPARQL) + NL | 混合查询 |
| **索引构建成本** | 低（仅需嵌入计算） | 高（需实体抽取+关系抽取+图谱构建） | 高 |
| **检索延迟** | 低（毫秒级） | 中（亚秒~秒级） | 中 |
| **可解释性** | 低（黑盒相似度） | 高（图路径即为证据链） | 高 |
| **冷启动难度** | 低 | 高（需领域本体设计） | 中 |
| **适用场景** | 事实型问答，FAQ | 多跳推理，实体关系，全局摘要 | 通用企业搜索 |
| **Schema要求** | 无 | 可选（本体/属性图模型） | 可选 |

---

## 12.2 知识图谱构建管道

### 12.2.1 构建流程总览

知识图谱的构建是GraphRAG中最关键、也最具工程挑战的环节。完整的构建管道如下：

```
+------------------------------------------------------------------+
|                  知识图谱构建管道 (KG Construction Pipeline)         |
+------------------------------------------------------------------+
|                                                                    |
|  [原始文档]                                                         |
|      |                                                             |
|      v                                                             |
|  +---------------------+                                           |
|  | Step 1: 文本预处理    |  分句、分词、篇章切分                        |
|  +---------------------+                                           |
|      |                                                             |
|      v                                                             |
|  +---------------------+    +------------------+                   |
|  | Step 2: 实体抽取      |--->| 实体消歧/链接     |                  |
|  | (NER/LLM/Span Detect)|    | (Entity Resolution)|                 |
|  +---------------------+    +------------------+                   |
|      |                             |                               |
|      v                             v                               |
|  +---------------------+    +------------------+                   |
|  | Step 3: 关系抽取      |    | Step 4: 属性抽取  |                  |
|  | (Relation Classify)  |    | (Property Extract)|                 |
|  +---------------------+    +------------------+                   |
|      |                             |                               |
|      v                             v                               |
|  +----------------------------------------------------------------+ |
|  | Step 5: 图谱写入 (Graph Construction)                           | |
|  | - 节点创建 (MERGE/CREATE)                                      | |
|  | - 边创建 (MERGE/CREATE)                                        | |
|  | - 属性写入 (SET)                                                | |
|  +----------------------------------------------------------------+ |
|      |                                                             |
|      v                                                             |
|  +---------------------+                                           |
|  | Step 6: 图谱质量评估  |  完整性、准确性、一致性检查                  |
|  +---------------------+                                           |
|      |                                                             |
|      v                                                             |
|  +---------------------+                                           |
|  | Step 7: 社区检测      |  Leiden/Louvain 算法                     |
|  +---------------------+                                           |
|      |                                                             |
|      v                                                             |
|  +---------------------+                                           |
|  | Step 8: 社区摘要生成  |  LLM为每个社区生成摘要文本                  |
|  +---------------------+                                           |
|                                                                    |
+------------------------------------------------------------------+
```

### 12.2.2 实体抽取

实体抽取是知识图谱构建的第一步，目标是从非结构化文本中识别出有意义的实体及其类型。

#### 方法一：基于NER模型的实体抽取

适用于通用领域（人名、地名、机构名等）的实体识别：

```python
import spacy
from typing import List, Dict, Tuple

class NERBasedExtractor:
    """基于命名实体识别的实体抽取器"""
    
    def __init__(self, model_name: str = "zh_core_web_trf"):
        """
        初始化NER模型
        - zh_core_web_trf: 基于Transformer的中文模型, 精度高但速度较慢
        - zh_core_web_sm:  轻量级中文模型, 速度快但精度较低
        """
        self.nlp = spacy.load(model_name)
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        从文本中抽取命名实体
        
        Args:
            text: 输入文本
            
        Returns:
            [{"text": "张三", "label": "PERSON", "start": 0, "end": 2}, ...]
        """
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": getattr(ent, '_confidence', None)
            })
        return entities

# 使用示例
extractor = NERBasedExtractor()
text = "2024年3月, 阿里巴巴集团宣布张勇辞任CEO, 由吴泳铭接任。"
entities = extractor.extract_entities(text)
# [{"text": "2024年3月", "label": "DATE", ...},
#  {"text": "阿里巴巴集团", "label": "ORG", ...},
#  {"text": "张勇", "label": "PERSON", ...},
#  {"text": "吴泳铭", "label": "PERSON", ...}]
```

#### 方法二：基于LLM的实体抽取

适用于领域特定实体、细粒度实体类型的识别：

```python
from openai import OpenAI
import json
from typing import List, Dict

class LLMBasedExtractor:
    """基于大语言模型的实体抽取器"""
    
    def __init__(self, client: OpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model
    
    def extract_entities(
        self, 
        text: str, 
        entity_types: List[str] = None
    ) -> List[Dict]:
        """
        使用LLM进行实体抽取
        
        Args:
            text: 输入文本
            entity_types: 自定义实体类型列表, 
                          如 ["企业", "人物", "产品", "技术", "日期", "金额"]
        
        Returns:
            实体列表, 每个实体包含名称、类型、描述
        """
        type_desc = ""
        if entity_types:
            type_desc = f"请仅抽取以下类型的实体: {', '.join(entity_types)}"
        
        prompt = f"""请从以下文本中抽取实体, 返回JSON格式。

{type_desc}

文本:
{text}

请返回如下JSON格式, 每个实体包含:
- name: 实体名称
- type: 实体类型
- description: 简要描述(基于上下文)
- mentions: 在文本中被提及的方式(列表)

示例输出:
[
  {{
    "name": "阿里巴巴集团",
    "type": "企业",
    "description": "中国电商和云计算巨头",
    "mentions": ["阿里巴巴集团", "阿里"]
  }}
]

请直接返回JSON数组, 不要有其他文字。"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("entities", result) if isinstance(result, dict) else result

# 使用示例
client = OpenAI()
extractor = LLMBasedExtractor(client)
text = "华为Mate 60 Pro搭载麒麟9000S芯片, 支持卫星通话功能。"
entities = extractor.extract_entities(
    text, 
    entity_types=["企业", "产品", "芯片", "功能", "技术"]
)
```

#### 方法三：Span-based实体检测

适用于需要精确边界检测的场景：

```python
class SpanBasedExtractor:
    """
    基于Span的实体检测器
    
    核心思想: 将实体抽取建模为序列标注问题, 
    使用BIO标记方案（B-实体开始, I-实体内部, O-非实体）
    """
    
    def __init__(self, model_path: str):
        """
        加载预训练的Span检测模型
        
        Args:
            model_path: 微调后的BERT/BERT-CRF模型路径
        """
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.id2label = self.model.config.id2label
    
    def predict(self, text: str) -> List[Dict]:
        """
        预测文本中的实体Span
        
        Returns:
            [{"text": "麒麟9000S", "type": "芯片", "start": 14, "end": 22}, ...]
        """
        import torch
        
        tokens = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**tokens)
            predictions = torch.argmax(outputs.logits, dim=-1)[0]
        
        # BIO解码, 将标签序列还原为实体Span
        entities = self._bio_decode(
            predictions, 
            tokens.tokens(), 
            self.id2label
        )
        return entities
    
    def _bio_decode(
        self, 
        predictions, 
        tokens, 
        id2label
    ) -> List[Dict]:
        """BIO标记解码为实体Span"""
        entities = []
        current_entity = None
        
        for idx, (pred, token) in enumerate(zip(predictions, tokens)):
            label = id2label[pred.item()]
            
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                entity_type = label[2:]  # 去掉 "B-" 前缀
                current_entity = {
                    "text": token,
                    "type": entity_type
                }
            elif label.startswith("I-") and current_entity:
                current_entity["text"] += token
            else:  # "O" 标签
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
```

### 12.2.3 关系抽取

#### 方法一：基于提示词的关系抽取

```python
class RelationExtractor:
    """基于LLM的关系抽取器"""
    
    def __init__(self, client: OpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model
    
    def extract_relations(
        self, 
        text: str, 
        entities: List[Dict],
        relation_types: List[str] = None
    ) -> List[Dict]:
        """
        从文本中抽取实体间的关系
        
        Args:
            text: 原始文本
            entities: 已抽取的实体列表
            relation_types: 预定义的关系类型, 如 ["任职于", "投资", "合作", "收购"]
            
        Returns:
            [{"subject": "张勇", "relation": "任职于", "object": "阿里巴巴", 
              "evidence": "...", "confidence": 0.95}, ...]
        """
        entity_str = "\n".join([
            f"- {e['name']} (类型: {e['type']})" for e in entities
        ])
        
        relation_instruction = ""
        if relation_types:
            relation_instruction = f"请仅使用以下关系类型: {', '.join(relation_types)}"
        else:
            relation_instruction = "请从文本中推断最合适的关系类型。"
        
        prompt = f"""从以下文本中抽取实体之间的语义关系。

已知实体:
{entity_str}

{relation_instruction}

文本:
{text}

请返回JSON数组, 每个关系包含:
- subject: 主体实体名称
- relation: 关系类型
- object: 客体实体名称
- direction: 关系方向 (单向/双向)
- evidence: 从文本中摘录的证据句子
- confidence: 置信度 (0.0-1.0)

直接返回JSON数组。"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result if isinstance(result, list) else result.get("relations", [])

# 使用示例
extractor = RelationExtractor(client)
text = "华为技术有限公司成立于1987年, 总部位于深圳, 创始人任正非。"
entities = [
    {"name": "华为技术有限公司", "type": "企业"},
    {"name": "任正非", "type": "人物"},
    {"name": "深圳", "type": "地点"},
    {"name": "1987年", "type": "时间"}
]
relations = extractor.extract_relations(text, entities)
```

#### 方法二：基于依存句法的关系抽取

适用于需要精确句法依据的场景（不需要LLM调用，速度快且确定性强）：

```python
class DependencyBasedRelationExtractor:
    """
    基于依存句法分析的关系抽取器
    
    通过分析句子的依存树结构, 识别主语-谓语-宾语(SVO)三元组
    """
    
    def __init__(self, model_name: str = "zh_core_web_trf"):
        self.nlp = spacy.load(model_name)
    
    def extract_svo_triples(self, text: str) -> List[Dict]:
        """
        抽取文本中的SVO三元组
        
        算法:
        1. 对每个句子进行依存句法分析
        2. 找到动词作为关系谓词
        3. 通过nsubj(名词主语)找到主体
        4. 通过dobj(直接宾语)找到客体
        
        Returns:
            [{"subject": "华为", "predicate": "收购", "object": "荣耀"}, ...]
        """
        doc = self.nlp(text)
        triples = []
        
        for sent in doc.sents:
            for token in sent:
                # 找到动词作为核心谓词
                if token.pos_ == "VERB":
                    subject = None
                    object_ = None
                    
                    # 查找主语 (nsubj) 和宾语 (dobj)
                    for child in token.children:
                        if child.dep_ == "nsubj":
                            subject = self._get_span(child)
                        elif child.dep_ == "dobj":
                            object_ = self._get_span(child)
                    
                    if subject and object_:
                        triples.append({
                            "subject": subject,
                            "predicate": token.lemma_,
                            "object": object_,
                            "sentence": sent.text
                        })
        
        return triples
    
    def _get_span(self, token) -> str:
        """获取包含修饰语的完整实体短语"""
        span_tokens = [t for t in token.subtree]
        span_tokens.sort(key=lambda t: t.i)
        return "".join([t.text for t in span_tokens])
```

### 12.2.4 实体消歧与链接

实体消歧（Entity Resolution / Entity Linking）是确保图谱质量的关键步骤，目标是将不同文本中对同一实体的不同称呼统一为规范形式。

```python
class EntityResolver:
    """
    实体消歧与链接器
    
    核心任务:
    1. 指代消歧: "苹果很好吃" vs "苹果发布了iPhone" -> 果实 vs 公司
    2. 名称标准化: "阿里巴巴" / "阿里" / "阿里巴巴集团" -> 同一实体
    3. 实体链接: 将提到的人物/组织链接到知识库中的规范实体
    """
    
    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model
        self.canonical_entities = {}  # 规范实体字典
    
    def resolve_entities(
        self, 
        entities: List[Dict], 
        context: str
    ) -> List[Dict]:
        """
        对实体列表进行消歧和标准化
        
        Args:
            entities: 原始实体列表
            context: 实体出现的上下文
            
        Returns:
            消歧后的实体列表, 每个实体增加了canonical_name字段
        """
        entity_names = [e["name"] for e in entities]
        
        prompt = f"""请对以下实体列表进行消歧和标准化。

上下文: {context}

实体列表: {', '.join(entity_names)}

请对每个实体进行:
1. 消歧: 判断该实体在上下文中的确切含义
2. 标准化: 将简称/别称转换为标准全称

返回JSON数组, 每个元素包含:
- original: 原始名称
- canonical: 标准名称
- disambiguation: 消歧说明
- should_merge_with: 列表中应合并的其他实体名称 (如果有的话)

直接返回JSON数组。"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        results = json.loads(response.choices[0].message.content)
        if isinstance(results, dict):
            results = results.get("entities", results.get("results", []))
        
        # 应用消歧结果
        resolved_map = {
            r["original"]: r["canonical"] for r in results
        }
        
        resolved_entities = []
        for entity in entities:
            canonical = resolved_map.get(entity["name"], entity["name"])
            entity["canonical_name"] = canonical
            resolved_entities.append(entity)
        
        return resolved_entities
```

### 12.2.5 图谱质量评估

图谱构建完成后，需要从多个维度评估质量：

| 维度 | 指标 | 计算方法 | 目标值 |
|------|------|---------|--------|
| **完整性** | 实体覆盖率 | 正确召回实体数 / 应召回实体总数 | > 90% |
| | 关系覆盖率 | 正确召回关系数 / 应召回关系总数 | > 85% |
| **准确性** | 实体精确率 | 正确抽取实体数 / 抽取实体总数 | > 95% |
| | 关系精确率 | 正确关系数 / 抽取关系总数 | > 90% |
| **一致性** | 实体合并率 | 成功合并的重复实体对 / 应合并的重复实体对 | > 95% |
| **连通性** | 孤立节点比例 | 度数为0的节点数 / 总节点数 | < 5% |
| **社区结构** | 模块度 | 社区检测的模块度分数 | > 0.3 |

```python
class GraphQualityAssessor:
    """知识图谱质量评估器"""
    
    def __init__(self, graph):
        self.graph = graph
    
    def assess_completeness(
        self, 
        ground_truth_entities: List[str], 
        extracted_entities: List[str]
    ) -> Dict:
        """评估实体覆盖率"""
        gt_set = set(ground_truth_entities)
        ex_set = set(extracted_entities)
        
        recall = len(gt_set & ex_set) / len(gt_set) if gt_set else 0
        precision = len(gt_set & ex_set) / len(ex_set) if ex_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        
        return {
            "recall": recall,
            "precision": precision, 
            "f1": f1,
            "missed_entities": list(gt_set - ex_set),
            "extra_entities": list(ex_set - gt_set)
        }
    
    def assess_connectivity(self) -> Dict:
        """评估图谱连通性"""
        isolated_nodes = [
            node for node in self.graph.nodes 
            if self.graph.degree(node) == 0
        ]
        return {
            "total_nodes": len(self.graph.nodes),
            "isolated_nodes": len(isolated_nodes),
            "isolation_ratio": len(isolated_nodes) / len(self.graph.nodes),
            "avg_degree": sum(dict(self.graph.degree()).values()) / len(self.graph.nodes)
        }
```

---

## 12.3 图数据库选型

### 12.3.1 Neo4j

Neo4j是目前最成熟、生态最完善的原生图数据库，在企业GraphRAG场景中被广泛采用。

**核心特性**：
- **查询语言**：Cypher（声明式图查询语言，ISO标准）
- **存储模型**：原生图存储（节点和边以邻接表形式物理存储）
- **图算法库**：GDS (Graph Data Science)库，内置社区检测、中心性分析、路径查找等60+算法
- **向量支持**：5.0+版本原生支持向量索引，可实现图+向量混合检索
- **部署方式**：社区版(免费)、企业版、AuraDB(云原生)

**在GraphRAG中的应用**：
```cypher
-- 创建实体节点
CREATE (e:Entity {
    id: "ent_001", 
    name: "华为技术有限公司", 
    type: "企业",
    description: "中国通信设备和智能终端制造商",
    embedding: [0.123, -0.456, ...]  -- 向量嵌入
})

-- 创建关系边
MATCH (a:Entity {id: "ent_001"}), (b:Entity {id: "ent_002"})
CREATE (a)-[r:COMPETES_WITH {
    since: 2010,
    evidence: "华为与中兴在通信设备领域竞争多年",
    confidence: 0.95
}]->(b)

-- 多跳查询: 查找与华为有合作关系的公司的CEO
MATCH (a:Entity {name: "华为技术有限公司"})-[r:COOPERATES_WITH]->(b:Entity)
MATCH (b)<-[s:CEO_OF]-(c:Entity {type: "人物"})
RETURN a.name, r.area, b.name, c.name

-- 使用GDS社区检测
CALL gds.graph.project('myGraph', 'Entity', {
    COOPERATES_WITH: {orientation: 'UNDIRECTED'},
    COMPETES_WITH: {orientation: 'UNDIRECTED'}
})
CALL gds.leiden.write('myGraph', {writeProperty: 'community_id'})
YIELD communityCount, modularity
```

### 12.3.2 NebulaGraph

NebulaGraph是国产分布式图数据库，采用**shared-nothing**架构，适合大规模图谱场景。

**核心特性**：
- **查询语言**：nGQL（类SQL风格的图查询语言），也支持OpenCypher
- **存储架构**：计算存储分离，元数据/图数据/索引三层独立扩展
- **分布式**：原生分布式设计，支持水平扩展至数百节点
- **性能**：高吞吐写入（百万级/秒），适合实时图谱更新

### 12.3.3 TigerGraph

TigerGraph以**深度链接分析**能力著称，适合需要大规模图计算的场景。

**核心特性**：
- **查询语言**：GSQL（图灵完备的图查询/计算语言）
- **存储引擎**：原生分布式图存储，支持实时深度遍历
- **图算法**：内置MLLib图算法库
- **核心优势**：支持10跳以上的深度遍历，适合复杂关系推理

### 12.3.4 综合对比

```
+------------------------------------------------------------------+
|                    图数据库综合对比矩阵                               |
+------------------------------------------------------------------+
|                |    Neo4j      |  NebulaGraph  |   TigerGraph    |
|----------------|---------------|---------------|-----------------|
| 查询语言        | Cypher        | nGQL/Cypher   | GSQL            |
| 存储模型        | 原生图         | 键值+图层       | 原生图           |
| 分布式          | 企业版支持      | 原生分布式      | 原生分布式        |
| 向量索引        | 5.0+原生支持   | 需外部集成      | 需外部集成        |
| 社区检测        | GDS库内置      | 需外部计算      | MLLib内置        |
| 水平扩展        | 有限           | 强大           | 强大             |
| 开源协议        | GPLv3         | Apache 2.0    | 专有             |
| 社区活跃度       | 极高           | 高             | 中               |
| 学习曲线        | 低             | 中             | 高               |
| 适合图谱规模     | <100亿边       | 百亿~千亿边     | >千亿边           |
| 中文支持        | 好             | 优秀(国产)      | 一般             |
+------------------------------------------------------------------+
```

### 12.3.5 选型决策树

```
需要图数据库?
  |
  +-- 图谱规模 < 100万节点?
  |     +-- 需要向量混合检索? --> Neo4j 社区版
  |     +-- 仅需图查询 --> Neo4j 或 轻量方案 (NetworkX + SQLite)
  |
  +-- 图谱规模 100万~1亿节点?
  |     +-- 企业有Neo4j经验? --> Neo4j 企业版
  |     +-- 国产化需求? --> NebulaGraph
  |     +-- 需要深度链接分析(>5跳)? --> TigerGraph
  |
  +-- 图谱规模 > 1亿节点?
        +-- 分布式是刚需? --> NebulaGraph
        +-- 深度计算是刚需? --> TigerGraph
        +-- 预算充足? --> Neo4j AuraDB 或 TigerGraph Cloud
```

### 12.3.6 图DB vs 属性图 vs RDF三元组存储

| 维度 | 属性图 (Property Graph) | RDF三元组存储 | 图数据库 (Native GraphDB) |
|------|------------------------|--------------|--------------------------|
| **代表产品** | Neo4j, JanusGraph | AllegroGraph, Virtuoso | Neo4j, TigerGraph, NebulaGraph |
| **数据模型** | 节点+边+属性(键值对) | 主体-谓词-客体 三元组 | 原生节点+边+属性 |
| **Schema** | 灵活，可选Schema约束 | OWL/RDFS本体约束 | 灵活，可选Schema |
| **查询语言** | Cypher, Gremlin | SPARQL | Cypher, GSQL, nGQL |
| **推理能力** | 弱(需外部集成) | 强(内置本体推理) | 弱~中 |
| **适用场景** | GraphRAG，企业知识图谱 | 语义网，本体推理，数据集成 | GraphRAG，实时图分析 |

**选择建议**：
- GraphRAG场景优先选择**属性图模型**（Neo4j），灵活且高效
- 需要本体推理和语义互操作的场景选择**RDF+SPARQL**
- 海量图谱选择**分布式原生图数据库**（NebulaGraph/TigerGraph）

---

## 12.4 GraphRAG检索方法

### 12.4.1 基于图遍历的检索

图遍历检索是最基础的GraphRAG检索方式，通过从查询中识别到的实体节点出发，沿关系边遍历获取相关上下文。

**BFS遍历策略**（广度优先，适合全局探索）：

```python
from collections import deque
from typing import Set, List, Dict, Any

class GraphTraversalRetriever:
    """基于图遍历的检索器"""
    
    def __init__(self, graph_driver):
        """
        Args:
            graph_driver: Neo4j driver 实例
        """
        self.driver = graph_driver
    
    def bfs_retrieve(
        self, 
        seed_entities: List[str], 
        max_depth: int = 2,
        max_nodes: int = 50,
        relation_filter: List[str] = None
    ) -> Dict[str, Any]:
        """
        BFS遍历检索
        
        算法:
        1. 从种子实体开始
        2. 逐层扩展邻居节点
        3. 记录遍历路径作为上下文
        
        Args:
            seed_entities: 种子实体名称列表
            max_depth: 最大遍历深度
            max_nodes: 最大收集节点数
            relation_filter: 关系类型过滤列表
        """
        # 构建关系过滤子句
        relation_clause = ""
        if relation_filter:
            rel_types = "|".join(relation_filter)
            relation_clause = f":{rel_types}"
        
        # Cypher BFS查询
        query = f"""
        MATCH path = (start:Entity)-[{relation_clause}*1..{max_depth}]-(neighbor:Entity)
        WHERE start.name IN $seed_names
        WITH path, 
             [node in nodes(path) | node.name] as entities,
             [rel in relationships(path) | type(rel)] as relations
        RETURN entities, relations, length(path) as depth
        LIMIT $max_nodes
        """
        
        with self.driver.session() as session:
            result = session.run(
                query,
                seed_names=seed_entities,
                max_nodes=max_nodes
            )
            
            paths = []
            entities_set = set()
            for record in result:
                paths.append({
                    "entities": record["entities"],
                    "relations": record["relations"],
                    "depth": record["depth"]
                })
                entities_set.update(record["entities"])
            
            return {
                "paths": paths,
                "unique_entities": list(entities_set),
                "path_count": len(paths)
            }
    
    def shortest_path_retrieve(
        self,
        entity_a: str,
        entity_b: str,
        max_depth: int = 5
    ) -> Dict:
        """最短路径检索 - 找出两个实体间的所有最短关系路径"""
        query = """
        MATCH path = shortestPath(
            (a:Entity {name: $name_a})-[*..%d]-(b:Entity {name: $name_b})
        )
        RETURN [node in nodes(path) | node.name] as entities,
               [rel in relationships(path) | type(rel)] as relations,
               length(path) as distance
        """ % max_depth
        
        with self.driver.session() as session:
            result = session.run(query, name_a=entity_a, name_b=entity_b)
            record = result.single()
            if record:
                return {
                    "entities": record["entities"],
                    "relations": record["relations"],
                    "distance": record["distance"],
                    "path_description": self._format_path(
                        record["entities"], record["relations"]
                    )
                }
            return {"path_description": "未找到路径"}
    
    def _format_path(self, entities: List[str], relations: List[str]) -> str:
        """将路径格式化为自然语言描述"""
        parts = [entities[0]]
        for i, rel in enumerate(relations):
            parts.append(f" --[{rel}]--> ")
            parts.append(entities[i + 1])
        return "".join(parts)
```

### 12.4.2 基于社区检测的检索

微软GraphRAG论文的核心创新之一是**社区检测+社区摘要**的分层检索策略。该策略先对整个文档图谱进行社区划分，为每个社区生成LLM摘要，然后在查询时选择最相关的社区摘要作为全局上下文。

```
+------------------------------------------------------------------+
|              社区检测检索流程 (Community-based Retrieval)            |
+------------------------------------------------------------------+
|                                                                    |
|  [文档图谱]                                                          |
|      |                                                             |
|      v                                                             |
|  +-----------------------+                                         |
|  | Leiden 社区检测算法    |                                         |
|  | - 模块度优化           |                                         |
|  | - 多层级划分           |                                         |
|  | Level 0: 细粒度社区    |                                         |
|  | Level 1: 中粒度社区    |                                         |
|  | Level 2: 粗粒度社区    |                                         |
|  +-----------------------+                                         |
|      |                                                             |
|      v                                                             |
|  +-----------------------+                                         |
|  | LLM 社区摘要生成      |                                         |
|  | 为每个社区生成:        |                                         |
|  | - 社区主题描述         |                                         |
|  | - 核心实体列表         |                                         |
|  | - 关键关系摘要         |                                         |
|  | - 代表性文本块         |                                         |
|  +-----------------------+                                         |
|      |                                                             |
|      v                                                             |
|  +-----------------------+                                         |
|  | 查询-社区匹配          |                                         |
|  | 1. 向量化查询          |                                         |
|  | 2. 计算查询与所有社区    |                                         |
|  |    摘要的相似度         |                                         |
|  | 3. 选择Top-K相关社区   |                                         |
|  +-----------------------+                                         |
|      |                                                             |
|      v                                                             |
|  [相关社区摘要作为LLM上下文]                                         |
|                                                                    |
+------------------------------------------------------------------+
```

**Leiden算法实现（基于Neo4j GDS）**：

```python
class CommunityBasedRetriever:
    """基于社区检测的GraphRAG检索器"""
    
    def __init__(self, graph_driver, llm_client):
        self.driver = graph_driver
        self.llm = llm_client
    
    def detect_communities(self) -> Dict:
        """
        执行Leiden社区检测算法
        
        Leiden算法相比Louvain的优势:
        - 保证社区连通性
        - 更快的收敛速度
        - 更好的模块度优化结果
        """
        with self.driver.session() as session:
            # Step 1: 创建内存图投影
            session.run("""
                CALL gds.graph.project(
                    'kg_graph',
                    'Entity',
                    {
                        CO_OCCURS: {orientation: 'UNDIRECTED'},
                        RELATED_TO: {orientation: 'UNDIRECTED'}
                    }
                )
            """)
            
            # Step 2: 运行Leiden算法
            result = session.run("""
                CALL gds.leiden.write('kg_graph', {
                    writeProperty: 'community_id',
                    includeIntermediateCommunities: true
                })
                YIELD communityCount, modularity, modularities, ranLevels
                RETURN communityCount, modularity, modularities, ranLevels
            """)
            
            record = result.single()
            
            # Step 3: 获取每个社区的信息
            communities = session.run("""
                MATCH (e:Entity)
                WHERE e.community_id IS NOT NULL
                RETURN e.community_id as community_id,
                       collect(e.name) as entities,
                       count(e) as size
                ORDER BY size DESC
            """)
            
            return {
                "community_count": record["communityCount"],
                "modularity": record["modularity"],
                "communities": [
                    {"id": c["community_id"], 
                     "entities": c["entities"], 
                     "size": c["size"]}
                    for c in communities
                ]
            }
    
    def generate_community_summaries(
        self, 
        communities: List[Dict]
    ) -> List[Dict]:
        """
        为每个社区生成LLM摘要
        
        微软GraphRAG论文的核心创新之一: 
        使用Map-Reduce模式, 先为每个社区生成独立摘要,
        再聚合为全局摘要
        """
        summaries = []
        
        for community in communities:
            entity_list = community["entities"][:20]  # 限制实体数量
            entity_str = ", ".join(entity_list)
            
            prompt = f"""请为以下知识图谱社区生成结构化摘要。

社区实体: {entity_str}

请生成:
1. 主题(Topic): 1-2句话描述社区主题
2. 核心实体(Key Entities): 最重要的3-5个实体
3. 关系摘要(Relationship Summary): 实体间的主要关系模式
4. 代表性问题(Representative Questions): 该社区能回答的3个典型问题

返回JSON格式。"""
            
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            summary = json.loads(response.choices[0].message.content)
            summary["community_id"] = community["id"]
            summary["entity_count"] = community["size"]
            summaries.append(summary)
        
        return summaries
    
    def retrieve_by_community(
        self, 
        query: str, 
        community_summaries: List[Dict],
        top_k: int = 3
    ) -> List[Dict]:
        """
        根据查询匹配最相关的社区
        
        使用向量相似度匹配查询与社区摘要
        """
        # Step 1: 向量化查询和所有社区摘要
        query_embedding = self._embed(query)
        
        scored_communities = []
        for summary in community_summaries:
            summary_text = f"{summary.get('topic', '')} {' '.join(summary.get('key_entities', []))}"
            summary_embedding = self._embed(summary_text)
            similarity = self._cosine_similarity(query_embedding, summary_embedding)
            scored_communities.append({
                "community": summary,
                "score": similarity
            })
        
        # Step 2: 排序并返回Top-K
        scored_communities.sort(key=lambda x: x["score"], reverse=True)
        return scored_communities[:top_k]
    
    def _embed(self, text: str) -> List[float]:
        """文本向量化"""
        # 使用OpenAI或其他嵌入模型
        response = self.llm.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """余弦相似度计算"""
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### 12.4.3 混合图+向量检索

这是工业级GraphRAG系统最常用的检索模式，将图结构的精确匹配与向量语义的模糊匹配相结合：

```python
class HybridGraphVectorRetriever:
    """
    混合图+向量检索器
    
    核心流水线:
    Query -> Entity Linking -> Graph Traversal -> Vector Filtering -> Reranking
    
    融合策略:
    1. 图权重: 基于图距离/边权重的相关性分数
    2. 向量权重: 基于余弦相似度的语义相关性分数
    3. 加权融合: score = alpha * graph_score + (1-alpha) * vector_score
    """
    
    def __init__(self, graph_driver, vector_index, alpha: float = 0.6):
        """
        Args:
            graph_driver: 图数据库驱动
            vector_index: 向量索引 (如 Pinecone, Milvus, Qdrant)
            alpha: 图分数权重, 0-1之间
        """
        self.graph = graph_driver
        self.vector_index = vector_index
        self.alpha = alpha
    
    def retrieve(
        self,
        query: str,
        query_entities: List[str],
        top_k: int = 10,
        graph_depth: int = 2
    ) -> List[Dict]:
        """
        混合检索主入口
        
        步骤:
        1. 图遍历获取结构相关节点
        2. 向量检索获取语义相关节点
        3. 融合排序
        """
        # Step 1: 图遍历检索
        graph_results = self._graph_retrieve(
            query_entities, 
            depth=graph_depth
        )
        
        # Step 2: 向量检索
        query_embedding = self._embed(query)
        vector_results = self._vector_retrieve(
            query_embedding, 
            top_k=top_k * 2  # 召回更多用于融合
        )
        
        # Step 3: 融合排序
        fused_results = self._fuse_and_rank(
            graph_results, 
            vector_results, 
            query_embedding,
            top_k
        )
        
        return fused_results
    
    def _graph_retrieve(
        self, 
        entities: List[str], 
        depth: int
    ) -> List[Dict]:
        """图遍历检索"""
        with self.graph.session() as session:
            result = session.run("""
                MATCH path = (start:Entity)-[*1..%d]-(neighbor:Entity)
                WHERE start.name IN $entities
                WITH neighbor, 
                     collect(distinct start.name) as sources,
                     min(length(path)) as min_distance
                RETURN neighbor.name as name,
                       neighbor.description as description,
                       sources,
                       min_distance,
                       1.0 / (min_distance + 1.0) as graph_score
                LIMIT 50
            """ % depth, entities=entities)
            
            return [dict(record) for record in result]
    
    def _vector_retrieve(
        self, 
        query_embedding: List[float], 
        top_k: int
    ) -> List[Dict]:
        """向量检索"""
        results = self.vector_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return [
            {
                "name": r.metadata.get("entity_name"),
                "description": r.metadata.get("description"),
                "vector_score": float(r.score)
            }
            for r in results
        ]
    
    def _fuse_and_rank(
        self,
        graph_results: List[Dict],
        vector_results: List[Dict],
        query_embedding: List[float],
        top_k: int
    ) -> List[Dict]:
        """
        融合排序
        
        使用加权融合:
        final_score = alpha * graph_score + (1-alpha) * vector_score
        
        同时应用互补增强:
        - 同时出现在图和向量结果中的节点获得boost
        - 仅在图结果中的节点保留基础图分数
        - 仅在向量结果中的节点保留惩罚后的向量分数
        """
        # 构建统一的结果映射
        node_scores = {}
        
        # 处理图结果
        for item in graph_results:
            name = item["name"]
            node_scores[name] = {
                "name": name,
                "description": item.get("description", ""),
                "graph_score": item["graph_score"],
                "vector_score": 0.0,
                "sources": item.get("sources", []),
                "source": "graph"
            }
        
        # 处理向量结果
        for item in vector_results:
            name = item["name"]
            vec_score = item["vector_score"]
            
            if name in node_scores:
                # 同时出现在两个结果中 -> Boost
                node_scores[name]["vector_score"] = vec_score
                node_scores[name]["source"] = "both"
            else:
                node_scores[name] = {
                    "name": name,
                    "description": item.get("description", ""),
                    "graph_score": 0.0,
                    "vector_score": vec_score * 0.8,  # 仅有向量分数的惩罚
                    "sources": [],
                    "source": "vector"
                }
        
        # 计算融合分数
        for name, scores in node_scores.items():
            if scores["source"] == "both":
                # 互补增强: 同时出现的节点获得额外boost
                boost = 1.2
            else:
                boost = 1.0
            
            scores["final_score"] = boost * (
                self.alpha * scores["graph_score"] + 
                (1 - self.alpha) * scores["vector_score"]
            )
        
        # 排序并返回Top-K
        ranked = sorted(
            node_scores.values(), 
            key=lambda x: x["final_score"], 
            reverse=True
        )
        
        return ranked[:top_k]
    
    def _embed(self, text: str) -> List[float]:
        """辅助: 文本嵌入"""
        # 实现略
        pass
```

### 12.4.4 多跳推理

多跳推理是GraphRAG区别于VectorRAG的核心能力，支持在知识图谱上进行链式推理：

```
+------------------------------------------------------------------+
|               多跳推理示例 (Multi-Hop Reasoning)                     |
+------------------------------------------------------------------+
|                                                                    |
|  查询: "与张勇合作过的人中, 哪些人现在是CEO?"                          |
|                                                                    |
|  Step 1: 实体链接 -> "张勇" (entity:e001)                           |
|                                                                    |
|  Step 2: 第1跳 - 找到合作者                                         |
|    张勇 --[合作]--> 马云                                            |
|    张勇 --[合作]--> 蔡崇信                                          |
|                                                                    |
|  Step 3: 第2跳 - 判断角色                                           |
|    马云 --[现任CEO_of]--> ? (无)                                    |
|    蔡崇信 --[现任CEO_of]--> 阿里巴巴集团 ✓                          |
|                                                                    |
|  Step 4: 返回答案: 蔡崇信 (现任阿里巴巴集团CEO)                       |
|                                                                    |
|  证据链: 张勇 -[合作]-> 蔡崇信 -[CEO_of]-> 阿里巴巴集团              |
|                                                                    |
+------------------------------------------------------------------+
```

```python
class MultiHopReasoner:
    """多跳推理器"""
    
    def __init__(self, graph_driver, llm_client):
        self.graph = graph_driver
        self.llm = llm_client
    
    def reason(
        self, 
        query: str, 
        max_hops: int = 3,
        max_paths: int = 5
    ) -> Dict:
        """
        多跳推理主流程
        
        策略: 迭代式推理
        1. LLM分析查询, 分解为推理步骤
        2. 每步执行图查询, 获取中间结果
        3. 基于中间结果执行下一步推理
        4. 最终聚合所有路径, 形成答案
        """
        # Step 1: LLM分析查询, 生成推理计划
        reasoning_plan = self._plan_reasoning(query, max_hops)
        
        # Step 2: 迭代执行推理步
        intermediate_results = []
        current_entities = reasoning_plan["seed_entities"]
        
        for hop in range(max_hops):
            if not current_entities:
                break
            
            # 执行当前跳的图查询
            hop_result = self._execute_hop(
                current_entities, 
                reasoning_plan["relation_types"][hop] 
                    if hop < len(reasoning_plan.get("relation_types", [])) 
                    else None
            )
            
            if not hop_result:
                break
            
            intermediate_results.append({
                "hop": hop + 1,
                "from_entities": current_entities,
                "results": hop_result
            })
            
            # 更新下一跳的起始实体
            current_entities = [
                r["next_entity"] for r in hop_result
            ]
        
        # Step 3: 聚合推理路径, 生成最终答案
        answer = self._synthesize_answer(
            query, 
            intermediate_results, 
            reasoning_plan
        )
        
        return {
            "query": query,
            "reasoning_plan": reasoning_plan,
            "intermediate_hops": intermediate_results,
            "answer": answer,
            "evidence_paths": self._extract_paths(intermediate_results)
        }
    
    def _plan_reasoning(self, query: str, max_hops: int) -> Dict:
        """LLM生成推理计划"""
        prompt = f"""分析以下查询, 生成多跳推理计划。

查询: {query}

请生成JSON格式的推理计划:
1. seed_entities: 查询中提到的实体名称列表
2. reasoning_steps: 推理步骤分解 (最多{max_hops}步)
   每个步骤包含:
   - description: 这一步要做什么
   - relation_types: 可能的关系类型
   - expected_output: 期望的输出类型
3. final_answer_type: 最终答案的预期类型

直接返回JSON。"""
        
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        plan = json.loads(response.choices[0].message.content)
        
        # 提取每步的关系类型
        relation_types = []
        for step in plan.get("reasoning_steps", []):
            relation_types.append(step.get("relation_types", []))
        
        return {
            "seed_entities": plan.get("seed_entities", []),
            "reasoning_steps": plan.get("reasoning_steps", []),
            "relation_types": relation_types
        }
    
    def _execute_hop(
        self, 
        entities: List[str], 
        relation_types: List[str]
    ) -> List[Dict]:
        """执行单跳图查询"""
        if not entities:
            return []
        
        rel_filter = ""
        if relation_types:
            rel_pattern = "|".join(relation_types)
            rel_filter = f":{rel_pattern}"
        
        query = f"""
        MATCH (a:Entity)-[r{rel_filter}]->(b:Entity)
        WHERE a.name IN $entities
        RETURN a.name as from_entity,
               type(r) as relation,
               b.name as next_entity,
               b.description as description,
               b.embedding as embedding
        LIMIT 20
        """
        
        with self.graph.session() as session:
            result = session.run(query, entities=entities)
            return [dict(record) for record in result]
    
    def _synthesize_answer(
        self, 
        query: str, 
        hop_results: List[Dict], 
        plan: Dict
    ) -> str:
        """综合推理结果生成最终答案"""
        # 构建推理历程文本
        reasoning_log = []
        for hop in hop_results:
            hop_text = f"第{hop['hop']}跳: 从 {hop['from_entities']} 出发, "
            hop_text += f"发现 {len(hop['results'])} 个结果: "
            hop_text += "; ".join([
                f"{r['from_entity']} -[{r['relation']}]-> {r['next_entity']}"
                for r in hop['results'][:5]
            ])
            reasoning_log.append(hop_text)
        
        prompt = f"""基于以下多跳推理历程, 回答原始查询。

查询: {query}

推理历程:
{chr(10).join(reasoning_log)}

请直接给出简洁准确的答案, 并引用推理路径中的证据。"""
        
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    def _extract_paths(self, hop_results: List[Dict]) -> List[List[str]]:
        """提取推理路径链"""
        paths = []
        if not hop_results:
            return paths
        
        # 构建路径: 第一跳的结果作为起始
        for r1 in hop_results[0]["results"]:
            path = [r1["from_entity"], r1["relation"], r1["next_entity"]]
            
            # 后续跳的结果追加
            for hop in hop_results[1:]:
                for r in hop["results"]:
                    if r["from_entity"] == path[-1]:
                        path.extend([r["relation"], r["next_entity"]])
                        break
            
            paths.append(path)
        
        return paths
```

### 12.4.5 图查询生成（Text2Cypher）

将自然语言问题自动翻译为图查询语言（如Cypher），是实现GraphRAG自助查询的关键技术：

```python
class Text2Cypher:
    """自然语言转Cypher查询"""
    
    def __init__(self, llm_client, graph_driver):
        self.llm = llm_client
        self.graph = graph_driver
        self.schema = self._load_schema()
    
    def _load_schema(self) -> str:
        """加载图谱Schema描述"""
        with self.graph.session() as session:
            # 获取节点标签
            labels = session.run("CALL db.labels()")
            node_types = [r["label"] for r in labels]
            
            # 获取关系类型
            rel_types = session.run("CALL db.relationshipTypes()")
            rels = [r["relationshipType"] for r in rel_types]
            
            schema = f"""
图谱Schema:
节点类型: {', '.join(node_types)}
关系类型: {', '.join(rels)}

节点属性: name(实体名称), type(实体类型), description(描述), embedding(向量)
关系属性: evidence(证据文本), confidence(置信度), since(起始时间)
"""
            return schema
    
    def generate_cypher(self, question: str) -> Dict:
        """
        将自然语言问题转换为Cypher查询
        
        返回生成的Cypher和执行结果
        """
        prompt = f"""你是一个Cypher查询生成专家。请将以下自然语言问题转换为Neo4j Cypher查询。

{self.schema}

问题: {question}

要求:
1. 使用MATCH语句匹配图模式
2. 使用参数化实体名称
3. 返回相关节点的name、description和关系类型
4. 限制最大返回条数为20
5. 确保查询是只读的 (不要包含CREATE/DELETE/MERGE)

请返回JSON格式:
{{
  "cypher": "MATCH ...",
  "explanation": "此Cypher查询的解释",
  "param_names": ["entity1", "entity2"]
}}

直接返回JSON。"""
        
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        cypher_result = json.loads(response.choices[0].message.content)
        
        # 验证Cypher安全性: 只允许只读操作
        cypher = cypher_result["cypher"].upper()
        forbidden = ["CREATE", "DELETE", "MERGE", "SET", "REMOVE", "DROP"]
        for keyword in forbidden:
            if keyword in cypher:
                raise ValueError(f"Cypher查询包含禁止的操作: {keyword}")
        
        # 执行Cypher查询
        with self.graph.session() as session:
            result = session.run(cypher_result["cypher"])
            records = [dict(record) for record in result]
        
        return {
            "cypher": cypher_result["cypher"],
            "explanation": cypher_result["explanation"],
            "results": records,
            "result_count": len(records)
        }

# 使用示例
t2c = Text2Cypher(llm_client, graph_driver)
result = t2c.generate_cypher("华为的投资方有哪些?")
# 生成: MATCH (a:Entity {name: '华为'})-[r:INVESTED_IN]->(b)
#       WHERE b.type = '投资机构'
#       RETURN b.name, b.description, r.confidence
```

---

## 12.5 GraphRAG使用场景

### 12.5.1 适用场景矩阵

| 场景 | 适用度 | 说明 | 典型查询示例 |
|------|--------|------|-------------|
| **多跳实体关系问答** | ★★★★★ | GraphRAG的核心优势场景 | "张三的直属上级的部门有哪些下属?" |
| **全局文档集摘要** | ★★★★★ | 社区检测+社区摘要提供全局视角 | "总结2024年Q1的所有项目报告主题" |
| **复杂约束推理** | ★★★★★ | 图查询语言支持精确的约束表达 | "找出与A公司有合作关系且在B公司董事会任职的人" |
| **实体为中心的问答** | ★★★★☆ | 实体图提供结构化上下文 | "华为Mate 60的技术规格和供应链情况" |
| **因果链分析** | ★★★★☆ | 通过关系链追踪因果 | "供应链中断对产品交付的影响路径" |
| **知识发现/探索** | ★★★★☆ | 图结构天然支持探索式分析 | "竞品A和B之间的技术关联有哪些?" |
| **简单事实型问答** | ★★☆☆☆ | VectorRAG更高效 | "珠穆朗玛峰的高度是多少?" |
| **高吞吐搜索** | ★★☆☆☆ | 图查询延迟高于向量检索 | 搜索引擎式的大规模文档检索 |

### 12.5.2 不适用场景

| 场景 | 原因 | 替代方案 |
|------|------|---------|
| **简单事实查询** | GraphRAG的构建成本远超VectorRAG, 但收益有限 | VectorRAG |
| **高吞吐实时搜索** | 图遍历是串行操作, 延迟难以降低到毫秒级 | 向量索引 + Elasticsearch |
| **领域本体缺失** | 无Schema的图谱质量难以保证 | 先构建领域本体, 再应用GraphRAG |
| **短文本/Tweet级内容** | 实体稀疏, 图结构不能提供有效信息增益 | VectorRAG + 关键词检索 |
| **低价值文档集** | 图谱构建的ROI不足 | 直接使用LLM |
| **频繁更新的内容源** | 增量更新图索引的成本高 | VectorRAG(增量嵌入更新更简单) |

### 12.5.3 企业应用场景

#### 法律文档分析

```
+------------------------------------------------------------------+
|              法律GraphRAG: 案例-法条-当事人关系图谱                   |
+------------------------------------------------------------------+
|                                                                    |
|   [案件A: 专利权纠纷]                                               |
|        |                                                           |
|        v                                                           |
|   原告: 华为技术 ---[起诉]---> 被告: 三星电子                         |
|        |                           |                               |
|        | 引用                       | 引用                          |
|        v                           v                               |
|   [专利法第65条]              [专利法第70条]                         |
|        |                           |                               |
|        +----------[相关判例]--------+                               |
|                          |                                         |
|                          v                                         |
|                   [类似案件B: 判决赔偿5000万]                        |
|                                                                    |
|  查询: "类似专利侵权案例中, 赔偿金额通常如何计算?"                      |
|   -> GraphRAG检索相关法条+判例+计算规则+历史金额                      |
|                                                                    |
+------------------------------------------------------------------+
```

#### 医疗知识图谱

```
+------------------------------------------------------------------+
|               医疗GraphRAG: 疾病-药物-症状-检查关系图谱                |
+------------------------------------------------------------------+
|                                                                    |
|   [疾病: 2型糖尿病]                                                 |
|        |                                                           |
|        +---症状---> 多饮, 多食, 体重下降                              |
|        |                                                           |
|        +---检查---> HbA1c, 空腹血糖, OGTT                           |
|        |                                                           |
|        +---药物---> 二甲双胍 -[禁忌]-> 肾功能不全                     |
|        |              |                                             |
|        |              +[联合用药]-> 西格列汀                          |
|        |                                                           |
|        +---并发症-->                                              |
|              |                                                     |
|              +---> 视网膜病变 -[检查]-> 眼底照相                       |
|              +---> 肾病 -[检查]-> 尿微量白蛋白                        |
|              +---> 神经病变 -[药物]-> 普瑞巴林                        |
|                                                                    |
+------------------------------------------------------------------+
```

#### 供应链分析

```
+------------------------------------------------------------------+
|              供应链GraphRAG: 多层供应商关系图谱                        |
+------------------------------------------------------------------+
|                                                                    |
|  [最终产品: iPhone 17]                                              |
|        |                                                           |
|        +---组装---> [富士康 (中国郑州)]                              |
|        |               |                                           |
|        |               +---芯片---> [台积电 (台湾)]                  |
|        |               |               |                           |
|        |               |               +---设备---> [ASML (荷兰)]   |
|        |               |               +---硅片---> [信越化学 (日本)]|
|        |               |                                           |
|        |               +---显示屏---> [三星显示 (韩国)]               |
|        |               +---电池---> [宁德时代 (中国)]                |
|        |                                                           |
|  查询: "如果台积电产能下降30%, 哪些产品受影响?"                        |
|   -> 从台积电节点向上游遍历所有依赖路径                               |
|                                                                    |
+------------------------------------------------------------------+
```

#### 金融风控分析

| 维度 | 传统方法 | GraphRAG方法 |
|------|---------|-------------|
| **关联交易检测** | 规则匹配, 易遗漏间接关联 | 图遍历检测多层关联路径 |
| **担保圈分析** | SQL多表JOIN, 性能差 | 图模式匹配, 毫秒级检测 |
| **实际控制人穿透** | 逐层查询, 深度受限 | 无限制深度遍历 |
| **上下游风险传导** | 手动分析, 无法量化 | 路径权重计算+风险传播模型 |
| **反洗钱** | 基于规则, 误报率高 | 图异常检测+社区发现 |

---

## 12.6 GraphRAG实现

### 12.6.1 完整GraphRAG管道实现

```python
"""
完整的GraphRAG管道实现

依赖安装:
pip install neo4j openai spacy networkx leidenalg python-louvain
python -m spacy download zh_core_web_trf
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

import spacy
from neo4j import GraphDatabase
from openai import OpenAI

# ============================================================
# 配置
# ============================================================
@dataclass
class GraphRAGConfig:
    """GraphRAG系统配置"""
    # Neo4j连接
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    
    # LLM配置
    llm_model: str = "gpt-4o"
    llm_model_mini: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    
    # 图谱构建
    chunk_size: int = 1500
    chunk_overlap: int = 200
    max_entities_per_chunk: int = 20
    max_relations_per_chunk: int = 15
    
    # 检索配置
    graph_traversal_depth: int = 2
    hybrid_alpha: float = 0.6  # 图分数权重
    top_k_retrieval: int = 10
    
    # 社区检测
    leiden_resolution: float = 1.0
    min_community_size: int = 3

# ============================================================
# 1. 文档处理与文本切分
# ============================================================
class DocumentProcessor:
    """文档预处理器"""
    
    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.nlp = spacy.load("zh_core_web_sm")
    
    def process_documents(
        self, 
        documents: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        处理文档列表
        
        Args:
            documents: [{"title": "...", "content": "...", "source": "..."}, ...]
            
        Returns:
            切分后的文本块列表
        """
        chunks = []
        
        for doc in documents:
            doc_chunks = self._split_document(doc)
            chunks.extend(doc_chunks)
        
        return chunks
    
    def _split_document(
        self, 
        document: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """基于句子边界的文档切分"""
        content = document["content"]
        doc = self.nlp(content)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_len = len(sent_text)
            
            if current_length + sent_len > self.config.chunk_size and current_chunk:
                chunks.append({
                    "text": " ".join(current_chunk),
                    "title": document.get("title", ""),
                    "source": document.get("source", ""),
                    "chunk_index": len(chunks)
                })
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sent_text)
            current_length += sent_len
        
        # 最后一个文本块
        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "title": document.get("title", ""),
                "source": document.get("source", ""),
                "chunk_index": len(chunks)
            })
        
        return chunks

# ============================================================
# 2. 图谱构建管道
# ============================================================
class KnowledgeGraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.client = OpenAI()
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        )
    
    def build_from_chunks(
        self, 
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        从文本块构建知识图谱
        
        完整流程:
        1. 实体抽取
        2. 关系抽取
        3. 实体消歧
        4. 图谱写入
        5. 图索引创建
        6. 社区检测
        """
        all_entities = []
        all_relations = []
        
        print(f"开始处理 {len(chunks)} 个文本块...")
        
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                print(f"  处理进度: {i}/{len(chunks)}")
            
            # Step 1+2: 实体和关系联合抽取
            result = self._extract_entities_and_relations(chunk["text"])
            
            # 关联到源文本块
            for entity in result.get("entities", []):
                entity["source_chunk"] = i
                entity["source_title"] = chunk.get("title", "")
                entity["source_text"] = chunk["text"][:100]
            for relation in result.get("relations", []):
                relation["source_chunk"] = i
            
            all_entities.extend(result.get("entities", []))
            all_relations.extend(result.get("relations", []))
        
        print(f"抽取实体: {len(all_entities)} 个")
        print(f"抽取关系: {len(all_relations)} 个")
        
        # Step 3: 实体消歧
        resolved_entities = self._resolve_entities(all_entities)
        print(f"消歧后实体: {len(resolved_entities)} 个")
        
        # Step 4: 写入Neo4j
        self._write_to_neo4j(resolved_entities, all_relations)
        print("图谱已写入Neo4j")
        
        # Step 5: 创建索引
        self._create_indexes()
        
        # Step 6: 社区检测
        community_result = self._detect_communities()
        print(f"社区检测完成: {community_result['community_count']} 个社区")
        
        return {
            "entity_count": len(resolved_entities),
            "relation_count": len(all_relations),
            "community_count": community_result["community_count"],
            "modularity": community_result["modularity"]
        }
    
    def _extract_entities_and_relations(
        self, 
        text: str
    ) -> Dict[str, List]:
        """
        联合抽取实体和关系
        
        使用LLM一次性抽取实体和关系, 减少API调用次数
        """
        prompt = f"""请从以下文本中抽取实体和关系, 返回JSON格式。

文本:
{text}

请抽取:
1. 实体: 包括人物、组织、地点、产品、技术、日期、金额等
2. 关系: 实体之间的语义关系

每个实体包含:
- name: 实体名称
- type: 实体类型
- description: 基于上下文的简短描述

每个关系包含:
- subject: 主体实体名称
- relation: 关系类型 (如: 任职于、合作、收购、投资、位于、开发、竞争)
- object: 客体实体名称
- evidence: 支持该关系的原文句子

返回JSON:
{{
  "entities": [
    {{"name": "...", "type": "...", "description": "..."}}
  ],
  "relations": [
    {{"subject": "...", "relation": "...", "object": "...", "evidence": "..."}}
  ]
}}

直接返回JSON。"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.llm_model_mini,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
                max_tokens=4000
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"  抽取失败: {e}")
            return {"entities": [], "relations": []}
    
    def _resolve_entities(
        self, 
        entities: List[Dict]
    ) -> List[Dict]:
        """
        批量实体消歧
        
        策略: 
        1. 首先基于名称相似度进行初步聚类
        2. 对相似度高的实体对, 使用LLM判断是否应合并
        """
        if len(entities) <= 1:
            return entities
        
        # 按实体名称排序并分组
        name_groups = defaultdict(list)
        for entity in entities:
            name_groups[entity["name"]].append(entity)
        
        # 如果所有名称都不同, 直接返回
        if len(name_groups) == len(entities):
            return entities
        
        # 对重复名称进行合并
        resolved = []
        for name, group in name_groups.items():
            if len(group) == 1:
                resolved.append(group[0])
            else:
                # 合并同名的多个实体
                merged = group[0].copy()
                merged["description"] = "; ".join(
                    set(e.get("description", "") for e in group if e.get("description"))
                )
                merged["source_chunks"] = [
                    e.get("source_chunk") for e in group 
                    if "source_chunk" in e
                ]
                resolved.append(merged)
        
        return resolved
    
    def _write_to_neo4j(
        self, 
        entities: List[Dict], 
        relations: List[Dict]
    ):
        """批量写入实体和关系到Neo4j"""
        
        with self.driver.session() as session:
            # 写入实体节点
            for entity in entities:
                entity_id = entity.get("name", "").replace("'", "\\'")
                entity_type = entity.get("type", "未知")
                description = entity.get("description", "").replace("'", "\\'")
                
                session.run("""
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type,
                        e.description = $description,
                        e.updated_at = datetime()
                """, {
                    "name": entity["name"],
                    "type": entity_type,
                    "description": description
                })
            
            # 写入关系边
            for rel in relations:
                session.run("""
                    MATCH (a:Entity {name: $subject})
                    MATCH (b:Entity {name: $object})
                    MERGE (a)-[r:RELATED {type: $relation}]->(b)
                    SET r.evidence = $evidence,
                        r.updated_at = datetime()
                """, {
                    "subject": rel["subject"],
                    "object": rel["object"],
                    "relation": rel["relation"],
                    "evidence": rel.get("evidence", "")
                })
    
    def _create_indexes(self):
        """创建图谱索引以优化查询性能"""
        with self.driver.session() as session:
            # 实体名称索引
            session.run(
                "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)"
            )
            # 实体类型索引
            session.run(
                "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)"
            )
            # 全文搜索索引 (Neo4j 5.x)
            try:
                session.run("""
                    CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
                    FOR (e:Entity) ON EACH [e.name, e.description]
                """)
            except Exception:
                pass  # 某些版本不支持
    
    def _detect_communities(self) -> Dict:
        """
        使用Leiden算法检测社区结构
        
        需要在Neo4j中安装GDS插件
        """
        with self.driver.session() as session:
            try:
                # 创建图投影
                session.run("""
                    CALL gds.graph.project(
                        'kg_community_graph',
                        'Entity',
                        {
                            RELATED: {
                                orientation: 'UNDIRECTED',
                                properties: {}
                            }
                        }
                    )
                """)
                
                # 运行Leiden算法
                result = session.run("""
                    CALL gds.leiden.write(
                        'kg_community_graph',
                        {
                            writeProperty: 'community_id',
                            randomSeed: 42
                        }
                    )
                    YIELD communityCount, modularity
                    RETURN communityCount, modularity
                """)
                
                record = result.single()
                return {
                    "community_count": record["communityCount"],
                    "modularity": record["modularity"]
                }
            except Exception as e:
                print(f"  社区检测需要Neo4j GDS插件: {e}")
                return {"community_count": 0, "modularity": 0}
    
    def close(self):
        """关闭数据库连接"""
        self.driver.close()

# ============================================================
# 3. 查询引擎
# ============================================================
class GraphRAGQueryEngine:
    """GraphRAG查询引擎"""
    
    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.client = OpenAI()
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        GraphRAG查询入口
        
        流程:
        1. 查询解析: 提取关键实体
        2. 图检索: 基于实体进行图遍历
        3. 上下文构建: 将图路径转化为文本上下文
        4. LLM推理: 基于上下文生成答案
        
        Args:
            question: 用户问题
            
        Returns:
            包含答案、证据路径、中间结果的字典
        """
        # Step 1: 提取查询中的实体
        query_entities = self._extract_query_entities(question)
        
        # Step 2: 图遍历检索
        graph_context = self._retrieve_graph_context(
            query_entities,
            depth=self.config.graph_traversal_depth
        )
        
        # Step 3: 构建LLM上下文
        llm_context = self._build_llm_context(question, graph_context)
        
        # Step 4: LLM推理
        answer = self._generate_answer(question, llm_context)
        
        return {
            "question": question,
            "query_entities": query_entities,
            "graph_context": graph_context,
            "answer": answer,
            "evidence_paths": graph_context.get("paths", [])
        }
    
    def _extract_query_entities(self, question: str) -> List[str]:
        """从查询中提取实体名称"""
        # 这里可以使用LLM提取, 也可以先用简单匹配
        # 简化实现: 使用Neo4j全文搜索匹配已知实体
        
        with self.driver.session() as session:
            # 先尝试全文搜索
            try:
                result = session.run("""
                    CALL db.index.fulltext.queryNodes('entity_fulltext', $question)
                    YIELD node, score
                    WHERE score > 0.5
                    RETURN node.name as name, score
                    LIMIT 10
                """, {"question": question})
                
                entities = [r["name"] for r in result]
                if entities:
                    return entities
            except Exception:
                pass
            
            # 回退: LLM提取
            prompt = f"""从以下问题中提取实体名称(人物、组织、地点、产品等)。

问题: {question}

请返回JSON数组, 每个元素为实体名称字符串。
直接返回JSON数组, 不要有其他文字。"""
            
            response = self.client.chat.completions.create(
                model=self.config.llm_model_mini,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            entities = json.loads(response.choices[0].message.content)
            if isinstance(entities, dict):
                entities = entities.get("entities", [])
            return entities
    
    def _retrieve_graph_context(
        self, 
        entities: List[str], 
        depth: int
    ) -> Dict:
        """图遍历检索上下文"""
        if not entities:
            return {"paths": [], "nodes": [], "edges": []}
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = (start:Entity)-[r:RELATED*1..%d]-(neighbor:Entity)
                WHERE start.name IN $entities
                WITH path,
                     [node in nodes(path) | 
                      {name: node.name, type: node.type, description: node.description}
                     ] as path_nodes,
                     [rel in relationships(path) | 
                      {type: rel.type, evidence: rel.evidence}
                     ] as path_edges,
                     length(path) as distance
                RETURN path_nodes, path_edges, distance
                ORDER BY distance
                LIMIT 20
            """ % depth, entities=entities)
            
            paths = []
            all_nodes = {}
            all_edges = []
            
            for record in result:
                path_nodes = record["path_nodes"]
                path_edges = record["path_edges"]
                distance = record["distance"]
                
                # 去重节点
                for node in path_nodes:
                    all_nodes[node["name"]] = node
                
                # 构建路径描述
                path_str = self._format_path(path_nodes, path_edges)
                paths.append({
                    "path": path_str,
                    "distance": distance,
                    "nodes": [n["name"] for n in path_nodes]
                })
                all_edges.extend(path_edges)
            
            return {
                "paths": paths,
                "nodes": list(all_nodes.values()),
                "edges": all_edges,
                "path_count": len(paths),
                "unique_entity_count": len(all_nodes)
            }
    
    def _format_path(
        self, 
        nodes: List[Dict], 
        edges: List[Dict]
    ) -> str:
        """将图路径格式化为自然语言字符串"""
        if not nodes:
            return ""
        
        parts = [f"{nodes[0].get('name', '?')}({nodes[0].get('type', '')})"]
        for i, edge in enumerate(edges):
            if i + 1 < len(nodes):
                parts.append(f" --[{edge.get('type', '?')}]--> ")
                parts.append(f"{nodes[i+1].get('name', '?')}({nodes[i+1].get('type', '')})")
        
        return "".join(parts)
    
    def _build_llm_context(
        self, 
        question: str, 
        graph_context: Dict
    ) -> str:
        """构建LLM推理上下文"""
        parts = []
        
        # 图谱路径信息
        if graph_context["paths"]:
            parts.append("## 知识图谱中的相关路径:\n")
            for i, path in enumerate(graph_context["paths"][:10], 1):
                parts.append(f"{i}. {path['path']} (距离: {path['distance']})\n")
        
        # 实体信息
        if graph_context["nodes"]:
            parts.append("\n## 相关实体描述:\n")
            for node in graph_context["nodes"][:15]:
                desc = node.get("description", "")
                if desc:
                    parts.append(f"- **{node['name']}** ({node.get('type', '')}): {desc}\n")
        
        return "\n".join(parts)
    
    def _generate_answer(
        self, 
        question: str, 
        context: str
    ) -> str:
        """基于图谱上下文生成答案"""
        prompt = f"""基于以下知识图谱上下文, 回答问题。请基于图谱中的事实进行推理, 不要编造信息。

{context}

问题: {question}

请给出简洁准确的答案。如果上下文不足以回答问题, 请明确说明。"""
        
        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def close(self):
        self.driver.close()

# ============================================================
# 4. 使用示例
# ============================================================
if __name__ == "__main__":
    # 初始化配置
    config = GraphRAGConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        chunk_size=1500,
        graph_traversal_depth=2
    )
    
    # 示例文档
    documents = [
        {
            "title": "华为2024年度报告",
            "content": "华为技术有限公司2024年实现营收7000亿元, 同比增长9.6%。"
                      "研发投入1600亿元, 占营收22.9%。轮值董事长孟晚舟表示...",
            "source": "annual_report_2024.pdf"
        },
        {
            "title": "芯片产业分析",
            "content": "麒麟9000S芯片由海思半导体设计, 采用7nm工艺。"
                      "该芯片搭载于华为Mate 60 Pro旗舰手机...",
            "source": "industry_analysis.pdf"
        }
    ]
    
    # 构建图谱
    processor = DocumentProcessor(config)
    chunks = processor.process_documents(documents)
    
    builder = KnowledgeGraphBuilder(config)
    build_result = builder.build_from_chunks(chunks)
    print(f"图谱构建完成: {json.dumps(build_result, ensure_ascii=False, indent=2)}")
    
    # 执行查询
    engine = GraphRAGQueryEngine(config)
    result = engine.query("华为2024年的研发投入是多少?")
    print(f"\n问题: {result['question']}")
    print(f"回答: {result['answer']}")
    print(f"证据路径数: {len(result['evidence_paths'])}")
    
    # 清理
    builder.close()
    engine.close()
```

### 12.6.2 与现有RAG系统的集成

```python
class UnifiedRAGSystem:
    """
    统一RAG系统: VectorRAG + GraphRAG 混合架构
    
    路由策略:
    - 简单事实查询 -> VectorRAG
    - 需要关联推理 -> GraphRAG
    - 不确定 -> 双路并行 + 结果融合
    """
    
    def __init__(
        self,
        vector_rag,      # 现有VectorRAG实例
        graph_rag,       # GraphRAG查询引擎
        llm_client
    ):
        self.vector_rag = vector_rag
        self.graph_rag = graph_rag
        self.llm = llm_client
    
    def query(self, question: str) -> Dict:
        """
        智能路由查询
        
        流程:
        1. LLM分析查询意图
        2. 根据意图路由到不同检索器
        3. 对复杂查询进行双路融合
        """
        # Step 1: 查询意图分析
        intent = self._analyze_intent(question)
        
        # Step 2: 路由决策
        if intent["type"] == "simple_fact":
            # 简单事实 -> VectorRAG
            result = self.vector_rag.query(question)
            result["route"] = "vector_rag"
            
        elif intent["type"] == "multi_hop":
            # 多跳推理 -> GraphRAG
            result = self.graph_rag.query(question)
            result["route"] = "graph_rag"
            
        elif intent["type"] == "complex":
            # 复杂混合查询 -> 双路融合
            vector_result = self.vector_rag.query(question)
            graph_result = self.graph_rag.query(question)
            
            result = self._fuse_results(
                question,
                vector_result,
                graph_result
            )
            result["route"] = "hybrid"
            
        else:
            # 默认: VectorRAG
            result = self.vector_rag.query(question)
            result["route"] = "vector_rag_default"
        
        return result
    
    def _analyze_intent(self, question: str) -> Dict:
        """分析查询意图"""
        prompt = f"""分析以下查询, 判断其类型。

查询: {question}

请判断:
- simple_fact: 简单事实查询 (如: "X是什么?", "Y有多少?")
- multi_hop: 需要多步推理的查询 (如: "X和Y有什么关系?", "谁通过Z与A关联?")
- complex: 需要全局视角或综合分析 (如: "总结...趋势", "分析...影响")

返回JSON: {{"type": "simple_fact|multi_hop|complex", "reason": "..."}}"""
        
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    
    def _fuse_results(
        self,
        question: str,
        vector_result: Dict,
        graph_result: Dict
    ) -> Dict:
        """融合VectorRAG和GraphRAG的结果"""
        # 构建综合上下文
        context = f"""
[向量检索结果]
{vector_result.get('answer', '')}

[知识图谱检索结果]
{graph_result.get('answer', '')}

[图谱证据路径]
{json.dumps(graph_result.get('evidence_paths', []), ensure_ascii=False)}
"""
        
        # LLM二次综合
        prompt = f"""基于两个检索系统的结果, 综合生成最终答案。

{context}

问题: {question}

请给出最完整准确的回答, 同时注明信息来源于哪个检索系统。"""
        
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        return {
            "question": question,
            "answer": response.choices[0].message.content,
            "vector_result": vector_result.get("answer", ""),
            "graph_result": graph_result.get("answer", ""),
            "evidence_paths": graph_result.get("evidence_paths", [])
        }
```

### 12.6.3 性能优化

| 优化维度 | 策略 | 预期效果 |
|---------|------|---------|
| **检索延迟** | 1. 图遍历深度限制 (depth <= 3)<br>2. BFS提前终止 (top-k满足即停止)<br>3. 路径缓存 (热门查询结果缓存) | P99延迟从 2s 降至 300ms |
| **图谱构建** | 1. LLM批量处理 (每次API调用处理多个文本块)<br>2. 增量更新 (仅处理新增/修改的文档)<br>3. 异步处理 (生产者-消费者模式) | 构建速度提升 5-10x |
| **存储优化** | 1. 边压缩 (同类关系合并)<br>2. 节点属性裁剪 (仅保留检索必要字段)<br>3. 分层存储 (热数据内存, 冷数据磁盘) | 存储成本降低 40-60% |
| **查询精度** | 1. 混合排序优化 alpha 调参<br>2. 查询改写 (Query Rewriting)<br>3. 负反馈学习 (记录用户反馈调权) | MRR 提升 15-25% |
| **成本控制** | 1. LLM调用分级 (小模型抽取实体, 大模型推理)<br>2. 嵌入缓存 (相同的文本块复用向量)<br>3. 社区摘要预计算 (离线生成, 在线匹配) | API成本降低 50-70% |

---

## 12.7 面试高频问题

### 理论基础

**Q1: GraphRAG和传统RAG的核心区别是什么？**

GraphRAG通过在文档集上构建知识图谱，将检索从"语义相似度匹配"提升为"结构化关系推理"。核心区别在于：VectorRAG将文档视为独立的向量点，检索的是与查询最相似的点；GraphRAG将文档中的实体和关系构建为图网络，检索的是实体之间的路径和社区结构。GraphRAG能够回答需要多步推理的问题（如"A和B之间有什么关系？"），而VectorRAG只能回答单步语义匹配的问题。

**Q2: Leiden算法和Louvain算法在社区检测中的区别？**

Leiden是Louvain的改进算法，主要改进包括：1) 保证社区的连通性（Louvain可能产生不连通社区）；2) 引入细化阶段（refinement phase），在移动节点后进行局部优化；3) 通常收敛更快，模块度优化结果更好。在GraphRAG中，Leiden是微软论文推荐的社区检测算法。

**Q3: 知识图谱构建中实体消歧的常用方法？**

1) 基于规则：编辑距离、Jaccard相似度等字符串匹配方法；2) 基于上下文：使用实体出现的上下文文本计算语义相似度；3) 基于知识库：将实体链接到已有知识库（如Wikidata）的规范实体；4) 基于LLM：直接使用LLM判断两个实体提及是否指代同一实体。实际生产系统中通常组合使用。

### 工程实践

**Q4: GraphRAG的索引构建成本如何控制？**

核心策略：1) 分级LLM调用：实体抽取用小型模型（gpt-4o-mini），关系抽取仅在必要时用大模型；2) 增量处理：维护变更文档列表，仅重新处理新增/修改的文档；3) 缓存与批处理：文本块嵌入和社区摘要预计算并缓存；4) 选择适当的粒度：不需要为所有类型文档构建完整图谱，仅对高价值、结构化程度高的文档构建图谱。

**Q5: 如何评估GraphRAG系统的效果？**

多维度评估：1) 端到端准确率：答案BLEU/ROUGE、人工评估的多跳推理准确率；2) 检索质量：图谱路径的精确率/召回率、社区匹配的NDCG；3) 图谱质量：实体抽取的F1、关系抽取的准确率、图的连通性指标；4) 系统性能：QPS、P50/P99延迟、Token消耗量。

**Q6: GraphRAG适用于什么规模的知识图谱？**

取决于所选图数据库。Neo4j社区版适用于百万级节点、千万级边；企业版和NebulaGraph适用于亿级节点；TigerGraph适用于百亿级以上。但实际生产建议：对于GraphRAG检索场景，1万-100万实体节点已经可以覆盖绝大多数企业知识库的需求。重点是图谱质量而非规模。

### 架构设计

**Q7: 如何在GraphRAG和VectorRAG之间做路由决策？**

典型的混合架构路由策略：1) 查询意图分类：使用LLM或分类器判断查询是否涉及实体关系、多步推理、全局摘要等需要图谱的场景；2) 实体密度检测：检测查询是否包含已知图谱实体，实体密度高的查询走GraphRAG；3) 结果融合：对于不确定性高的查询，同时执行双路检索并将结果融合。建议默认使用VectorRAG，仅当检测到关系推理需求时切换到GraphRAG。

**Q8: GraphRAG的检索延迟如何优化？**

关键优化点：1) 限制图遍历深度（通常2-3跳足够覆盖大多数推理需求）；2) 使用索引加速：为实体名称、类型创建B-tree索引，为高频关系创建关系索引；3) 结果缓存：热门查询的图路径结果缓存（TTL策略）；4) 预计算：社区检测和社区摘要离线预计算，在线仅做向量匹配；5) 读写分离：图数据库的主从架构，检索走只读副本。

---

## 12.8 企业最佳实践

### 12.8.1 渐进式采用策略

```
+------------------------------------------------------------------+
|                GraphRAG 渐进式采用路线图                             |
+------------------------------------------------------------------+
|                                                                    |
|  阶段1: 评估期 (1-2周)                                              |
|  +-- 选择1-2个高价值试点场景 (如法律文档分析)                         |
|  +-- 使用Neo4j社区版 + 开源LLM搭建原型                              |
|  +-- 评估ROI: 检索精度提升 vs 构建成本                              |
|                                                                    |
|  阶段2: 试运行期 (2-4周)                                            |
|  +-- 在试点场景上构建完整GraphRAG管道                                |
|  +-- 与传统VectorRAG并行运行, A/B测试                               |
|  +-- 收集用户反馈和性能指标                                         |
|                                                                    |
|  阶段3: 正式部署 (4-8周)                                            |
|  +-- 根据A/B测试结果确定GraphRAG的适用场景                           |
|  +-- 部署统一路由架构 (VectorRAG + GraphRAG)                        |
|  +-- 建立监控和告警体系                                              |
|                                                                    |
|  阶段4: 持续优化                                                     |
|  +-- 图谱质量持续改进 (反馈循环)                                     |
|  +-- 增量更新管道自动化                                              |
|  +-- 根据使用模式调整基础设施规模                                    |
|                                                                    |
+------------------------------------------------------------------+
```

### 12.8.2 领域本体设计原则

1. **从业务出发**：本体设计应反映业务领域的实际概念和关系，而非技术抽象
2. **精简原则**：从最核心的5-10种实体类型和10-20种关系类型开始，随需求增长
3. **统一命名规范**：实体类型、关系类型的命名约定，如实体用名词（Person, Company），关系用动词（WORKS_AT, OWNS）
4. **属性最小化**：初始阶段仅存储检索必需的属性，避免属性膨胀
5. **可扩展性**：预留属性字段以支持未来扩展，但不要过度设计

### 12.8.3 数据安全与合规

- **敏感实体脱敏**：对涉及PII的实体（如人名、身份证号）进行哈希或脱敏处理后再入图
- **访问控制**：利用图数据库的基于角色的访问控制（RBAC），限制对特定标签/关系的访问
- **审计日志**：记录所有图查询操作，支持数据溯源和合规审计
- **数据生命周期**：设定图谱数据的保留策略，定期清理过期或无效的实体和关系

### 12.8.4 持续质量改进

```
+------------------------------------------------------------------+
|               图谱质量持续改进循环                                    |
+------------------------------------------------------------------+
|                                                                    |
|         +-------------+                                            |
|         | 用户反馈收集  | <-----+                                    |
|         +-------------+       |                                    |
|               |               |                                    |
|               v               |                                    |
|         +-------------+       |                                    |
|         | 质量问题分析  |       |                                    |
|         +-------------+       |                                    |
|               |               |                                    |
|               v               |                                    |
|         +-------------+       |                                    |
|         | 图谱修复策略  |       |                                    |
|         +-------------+       |                                    |
|               |               |                                    |
|          +----+----+          |                                    |
|          |         |          |                                    |
|          v         v          |                                    |
|   +-----------+ +----------+  |                                    |
|   | 自动修复  | | 人工审核  |--+                                    |
|   +-----------+ +----------+                                       |
|          |         |                                               |
|          +----+----+                                               |
|               |                                                    |
|               v                                                    |
|         +-------------+                                            |
|         | 效果验证     |                                            |
|         +-------------+                                            |
|                                                                    |
+------------------------------------------------------------------+
```

### 12.8.5 关键警告与常见陷阱

| 陷阱 | 后果 | 预防措施 |
|------|------|---------|
| **在不需要的文档上构建图谱** | 浪费计算资源, ROI低下 | 仅对结构化程度高、有实体关系的文档构建图谱 |
| **无限增长的知识图谱** | 检索性能下降, 存储成本膨胀 | 设定图谱规模上限, 实现数据生命周期管理 |
| **忽略图谱质量** | 错误的关系导致错误推理 | 建立质量监控仪表板, 定期人工抽查 |
| **过度依赖LLM抽取** | 成本高昂, 不稳定 | 组合使用NER模型+规则+LLM, 分层处理 |
| **冷启动时一步到位** | 项目失败风险高 | 渐进式采用, 先在最小可行场景验证 |
| **忽略图模式（Schema）设计** | 图谱难以查询和维护 | 先设计核心本体, 再开始构建数据 |
| **使用错误的关系方向** | 语义错误, 推理失败 | 关系命名要体现方向性, 如 OWNED_BY vs OWNS |

---

## 12.9 本章小结

GraphRAG将知识图谱的结构化语义理解引入RAG管道，实现了从"语义相似度检索"到"结构化关系推理"的范式升级。本章系统介绍了GraphRAG的基础原理、知识图谱构建管道、图数据库选型、检索方法、使用场景以及完整实现代码。

**核心要点回顾**：

1. **GraphRAG的核心价值**在于显式建模实体间的关系，支持多跳推理和全局摘要，弥补了VectorRAG在复杂推理场景中的不足
2. **知识图谱构建**是GraphRAG中最关键的工程环节，需要组合使用NER、LLM抽取、依存句法分析等技术，并辅以实体消歧和质量评估
3. **图数据库选型**应根据图谱规模、查询模式、部署约束综合考量：小型选Neo4j社区版，国产化选NebulaGraph，超大规模选TigerGraph
4. **混合图+向量检索**是当前工业界的最优实践，通过图结构精确匹配与向量语义模糊匹配的加权融合，实现检索精度和覆盖面的最佳平衡
5. **社区检测+社区摘要**是微软GraphRAG论文的核心创新，为文档集的全局理解提供了分层视角
6. **渐进式采用**是企业导入GraphRAG的最佳策略，从单一高价值场景出发，验证ROI后再逐步扩展

**与后续章节的关系**：

- 第13章将介绍Agentic RAG，其中智能体可以使用GraphRAG作为其工具之一
- 第14章将讨论RAG评估体系，GraphRAG需要专门的评估指标（如多跳推理准确率、图路径质量等）
- 第15章将介绍生产级RAG系统的部署与运维

---

> **推荐阅读**：
> - Microsoft Research, "GraphRAG: From Local to Global" (2024)
> - Neo4j Graph Data Science Documentation
> - Robinson, Webber, Eifrem, "Graph Databases" (O'Reilly)
> - Hogan et al., "Knowledge Graphs" (ACM Computing Surveys, 2021)
