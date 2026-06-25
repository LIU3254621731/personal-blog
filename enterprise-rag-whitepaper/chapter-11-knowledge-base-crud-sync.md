# 第11章 知识库CRUD与同步机制

> **核心命题**：在企业生产环境中，如何在原始文档、向量数据库、缓存、索引、关系型数据库之间维护一致性？

## 目录

- [11.1 文档生命周期操作](#111-文档生命周期操作)
- [11.2 一致性机制深度剖析](#112-一致性机制深度剖析)
- [11.3 防腐蚀机制](#113-防腐蚀机制)
- [11.4 完整同步架构](#114-完整同步架构)
- [11.5 实现设计](#115-实现设计)
- [11.6 面试题精选](#116-面试题精选)
- [11.7 企业最佳实践总结](#117-企业最佳实践总结)

---

## 11.1 文档生命周期操作

### 11.1.1 概念定义

知识库CRUD（Create/Read/Update/Delete）是RAG系统的基础操作层。与传统的数据库CRUD不同，RAG系统的CRUD涉及 **多存储介质联动**：
文档入库后，经过解析、分块、向量化、索引构建、缓存预热，数据以不同形态分布在多个存储层中。任何单一层的变更若不同步到其他层，就会产生一致性问题。

### 11.1.2 背景与核心问题

在RAG系统的实际运营中，以下场景频繁发生：

| 场景 | 挑战 | 影响 |
|------|------|------|
| 用户删除文档 | 向量残留导致"幽灵检索" | 检索到已删除内容的向量 |
| 用户更新文档 | 新旧向量共存 | 检索结果包含过期信息 |
| 批量导入 | 部分失败导致不一致 | 知识库状态不可知 |
| 并发更新 | 写写冲突 | 数据损坏或丢失 |

**核心问题**：如何保证操作在多存储层的 **原子性** 和 **最终一致性**？

### 11.1.3 文档删除流程

文档删除并非简单的"删一条记录"，而是一个 **级联清理** 过程。

#### 工作数据流

```
用户请求删除 doc_id=123
        │
        ▼
┌───────────────────┐
│  API Gateway      │  接收请求，鉴权
│  DELETE /docs/123 │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  Service Layer    │  开启事务
│  delete_doc(123)  │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐     ┌───────────────────┐
│  1. 查询所有块ID  │────▶│  关系型数据库      │
│  SELECT chunk_id  │     │  chunks 表         │
│  WHERE doc_id=123 │     │  GET: [c1,c2,c3]  │
└───────┬───────────┘     └───────────────────┘
        │
        ▼
┌───────────────────┐     ┌───────────────────┐
│  2. 删除所有向量  │────▶│  向量数据库        │
│  DELETE vectors   │     │  WHERE chunk_id    │
│  FOR c1,c2,c3     │     │  IN (c1,c2,c3)    │
└───────┬───────────┘     └───────────────────┘
        │
        ▼
┌───────────────────┐     ┌───────────────────┐
│  3. 删除文本块    │────▶│  关系型数据库      │
│  DELETE FROM      │     │  chunks 表         │
│  chunks WHERE     │     │  软删除或硬删除    │
│  doc_id=123       │     │                    │
└───────┬───────────┘     └───────────────────┘
        │
        ▼
┌───────────────────┐     ┌───────────────────┐
│  4. 失效缓存      │────▶│  Redis             │
│  DEL cache:doc:123│     │  DEL cache:chunk:* │
│  DEL cache:search:*│     │  模式匹配删除     │
└───────┬───────────┘     └───────────────────┘
        │
        ▼
┌───────────────────┐     ┌───────────────────┐
│  5. 更新全文索引  │────▶│  Elasticsearch     │
│  DELETE /idx/     │     │  异步删除索引文档  │
│  docs/_doc/123    │     │                    │
└───────┬───────────┘     └───────────────────┘
        │
        ▼
┌───────────────────┐
│  6. 发送变更事件  │────▶  Kafka topic:
│  Event: DOC_DEL   │      knowledgebase.events
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  7. 确认删除      │────▶  返回 200 OK
│  返回操作结果     │      {status: "deleted",
└───────────────────┘       affected_chunks: 3}
```

#### 核心实现代码

```python
"""
文档删除服务 —— 级联删除实现
"""
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DeleteMode(Enum):
    SOFT = "soft"      # 软删除，可恢复
    HARD = "hard"      # 硬删除，不可恢复


@dataclass
class DeleteResult:
    """删除操作结果"""
    doc_id: str
    status: str
    affected_chunks: int
    deleted_vectors: int
    cache_keys_invalidated: int
    index_docs_removed: int
    errors: List[str]


class DocumentDeleteService:
    """
    级联删除服务

    保证删除的原子性：如果任一步骤失败，整个操作回滚（软删除）
    或标记为部分失败（硬删除 + 补偿任务）。
    """

    def __init__(
        self,
        relational_db,       # PostgreSQL/MySQL
        vector_db,           # Milvus/Qdrant/Weaviate
        cache_client,        # Redis
        search_index,        # Elasticsearch
        event_bus,           # Kafka
    ):
        self.db = relational_db
        self.vector_db = vector_db
        self.cache = cache_client
        self.search_index = search_index
        self.event_bus = event_bus

    def delete_document(
        self,
        doc_id: str,
        mode: DeleteMode = DeleteMode.HARD
    ) -> DeleteResult:
        """
        删除文档及其所有关联数据

        Args:
            doc_id: 文档ID
            mode: 删除模式（软删除/硬删除）

        Returns:
            DeleteResult: 操作结果
        """
        errors: List[str] = []
        result = DeleteResult(
            doc_id=doc_id,
            status="pending",
            affected_chunks=0,
            deleted_vectors=0,
            cache_keys_invalidated=0,
            index_docs_removed=0,
            errors=errors,
        )

        # Step 1: 查询文档是否存在及其所有块
        doc = self.db.find_one("documents", {"id": doc_id})
        if not doc:
            raise DocumentNotFoundError(f"Document {doc_id} not found")

        chunks = self.db.find("chunks", {"doc_id": doc_id})
        chunk_ids = [c["id"] for c in chunks]
        result.affected_chunks = len(chunk_ids)

        if not chunk_ids:
            logger.warning(f"Document {doc_id} has no chunks")
            # 直接删除文档记录
            self._delete_doc_record(doc_id, mode)
            result.status = "deleted"
            return result

        # Step 2: 删除向量数据库中的向量
        try:
            self.vector_db.delete_by_filter(
                filter_expr=f'doc_id == "{doc_id}"'
            )
            result.deleted_vectors = len(chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} vectors for doc {doc_id}")
        except Exception as e:
            errors.append(f"Vector deletion failed: {e}")
            logger.error(f"Vector deletion error for doc {doc_id}: {e}")

        # Step 3: 删除关系数据库中的块记录
        try:
            if mode == DeleteMode.SOFT:
                self.db.update(
                    "chunks",
                    {"doc_id": doc_id},
                    {"status": "deleted", "deleted_at": "NOW()"}
                )
            else:
                self.db.delete("chunks", {"doc_id": doc_id})
        except Exception as e:
            errors.append(f"Chunk deletion failed: {e}")

        # Step 4: 失效缓存
        try:
            cache_keys = self._invalidate_cache(doc_id, chunk_ids)
            result.cache_keys_invalidated = len(cache_keys)
        except Exception as e:
            errors.append(f"Cache invalidation failed: {e}")

        # Step 5: 从搜索引擎索引中删除
        try:
            self.search_index.delete_by_query(
                index="documents",
                query={"term": {"doc_id": doc_id}}
            )
            result.index_docs_removed = len(chunk_ids)
        except Exception as e:
            errors.append(f"Search index deletion failed: {e}")

        # Step 6: 删除（或标记）文档记录
        self._delete_doc_record(doc_id, mode)

        # Step 7: 发送变更事件
        try:
            self.event_bus.publish(
                topic="knowledgebase.events",
                key=doc_id,
                value={
                    "event_type": "DOC_DELETED",
                    "doc_id": doc_id,
                    "mode": mode.value,
                    "timestamp": "NOW()",
                }
            )
        except Exception as e:
            # 事件发布失败不应阻断主流程
            logger.warning(f"Event publish failed for doc {doc_id}: {e}")

        # 如果有错误，触发补偿任务
        if errors:
            result.status = "partial_failure"
            self._schedule_compensation(doc_id, errors)
        else:
            result.status = "deleted"

        return result

    def _invalidate_cache(
        self, doc_id: str, chunk_ids: List[str]
    ) -> List[str]:
        """失效所有相关缓存键"""
        keys = []
        patterns = [
            f"doc:{doc_id}:*",           # 文档缓存
            f"chunk:{doc_id}:*",         # 块缓存
        ]
        for chunk_id in chunk_ids:
            patterns.append(f"embedding:{chunk_id}")

        for pattern in patterns:
            matched = self.cache.scan_keys(pattern)
            for key in matched:
                self.cache.delete(key)
                keys.append(key)

        # 失效搜索缓存（基于前缀）
        search_cache_keys = self.cache.scan_keys("search:*")
        for key in search_cache_keys:
            self.cache.delete(key)
            keys.append(key)

        return keys

    def _delete_doc_record(self, doc_id: str, mode: DeleteMode):
        """删除或标记文档记录"""
        if mode == DeleteMode.SOFT:
            self.db.update(
                "documents",
                {"id": doc_id},
                {
                    "status": "deleted",
                    "deleted_at": "NOW()",
                    "version": self.db.raw("version + 1"),
                }
            )
        else:
            self.db.delete("documents", {"id": doc_id})

    def _schedule_compensation(self, doc_id: str, errors: List[str]):
        """调度补偿任务以修复部分失败"""
        self.db.insert("compensation_tasks", {
            "task_type": "CASCADE_DELETE_RETRY",
            "target_id": doc_id,
            "errors": errors,
            "status": "pending",
            "retry_count": 0,
            "max_retries": 3,
            "next_retry_at": "NOW() + INTERVAL '30 seconds'",
        })
```

### 11.1.4 文档更新流程

更新操作的核心挑战是 **增量处理**：只重新处理变更的部分，避免全量重建。

#### 更新检测与增量处理架构

```
文档更新请求 PUT /docs/123 {content: "新内容"}
        │
        ▼
┌─────────────────────────────────────────────┐
│          Step 1: 差异检测 (Delta Detection)  │
├─────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐         │
│  │ 旧文档内容   │ vs │ 新文档内容   │         │
│  │ (从PG读取)   │    │ (请求体)     │         │
│  └──────┬──────┘    └──────┬──────┘         │
│         │                  │                 │
│         └────────┬─────────┘                 │
│                  ▼                           │
│    ┌──────────────────────────┐              │
│    │  Diff 算法 (difflib)     │              │
│    │  输出: 变更区域列表      │              │
│    │  [ {line: 5, op: "mod"}, │              │
│    │    {line: 12, op: "add"},│              │
│    │    {line: 20, op: "del"}]│              │
│    └──────────────────────────┘              │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│     Step 2: 选择性重分块 (Selective Re-chunk)│
├─────────────────────────────────────────────┤
│  ┌──────────────────────────────────────┐    │
│  │ 仅对变更区域及其上下文窗口重新分块   │    │
│  │                                     │    │
│  │ 受影响块: [chunk_2, chunk_3]        │    │
│  │ 新块内容: [...]                     │    │
│  │ 未受影响块: [chunk_1, chunk_4, ...] │    │
│  │  → 保持不变，不重新向量化            │    │
│  └──────────────────────────────────────┘    │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│    Step 3: 增量向量更新 (Incremental Vector) │
├─────────────────────────────────────────────┤
│  受影响块 → Embedding Model → 新向量         │
│  ┌──────────────────────────────────────┐    │
│  │ UPSERT INTO vectors                  │    │
│  │ (chunk_id, embedding, version)       │    │
│  │ VALUES                               │    │
│  │   ('chunk_2', [0.1, 0.2, ...], v2), │    │
│  │   ('chunk_3', [0.3, 0.4, ...], v2)  │    │
│  │ ON CONFLICT (chunk_id)               │    │
│  │ DO UPDATE SET embedding=EXCLUDED...  │    │
│  └──────────────────────────────────────┘    │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│     Step 4: 缓存刷新                         │
├─────────────────────────────────────────────┤
│  失效受影响块的缓存 + 搜索缓存                │
│  可选: 预测性预热新块的缓存                   │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
                 确认完成
```

#### 增量更新核心代码

```python
"""
文档增量更新服务
"""
import hashlib
import difflib
from typing import List, Set, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class ChunkInfo:
    """块信息"""
    chunk_id: str
    content: str
    start_line: int
    end_line: int
    embedding_hash: str = ""


@dataclass
class DiffRegion:
    """差异区域"""
    start_line: int
    end_line: int
    operation: str  # "add", "delete", "modify"


class DocumentUpdateService:
    """
    增量更新服务

    设计原则：
    1. 仅重建受变更影响的块
    2. 上下文窗口管理：如果块边界附近的内容变化了，相邻块也可能需要重建
    3. 版本追踪：每次更新递增版本号，支持回滚
    """

    CONTEXT_WINDOW_SIZE = 2  # 变更区域前后各包含2个块

    def __init__(self, embedding_model, vector_db, cache, event_bus):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.cache = cache
        self.event_bus = event_bus

    def update_document(
        self,
        doc_id: str,
        new_content: str,
        old_content: str,
        old_chunks: List[ChunkInfo],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> Dict:
        """
        增量更新文档

        Args:
            doc_id: 文档ID
            new_content: 新文档内容
            old_content: 旧文档内容
            old_chunks: 旧分块列表
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小

        Returns:
            更新结果字典
        """
        # Step 1: 计算内容哈希，判断是否真的变化了
        new_hash = hashlib.sha256(new_content.encode()).hexdigest()
        old_hash = hashlib.sha256(old_content.encode()).hexdigest()

        if new_hash == old_hash:
            logger.info(f"Document {doc_id} has no actual changes")
            return {"status": "unchanged", "doc_id": doc_id}

        # Step 2: 检测差异区域
        diff_regions = self._detect_diff_regions(old_content, new_content)

        # Step 3: 识别受影响的块
        affected_chunks, unaffected_chunks = self._identify_affected_chunks(
            diff_regions, old_chunks
        )

        logger.info(
            f"Document {doc_id}: {len(affected_chunks)} affected, "
            f"{len(unaffected_chunks)} unaffected chunks"
        )

        # Step 4: 仅对受影响区域重新分块
        new_affected_chunks = self._rechunk_affected_regions(
            new_content, diff_regions, chunk_size, chunk_overlap
        )

        # Step 5: 仅对变更块生成向量
        chunks_to_embed = [
            c for c in new_affected_chunks
            if c.embedding_hash != self._get_existing_hash(c.chunk_id)
        ]

        if chunks_to_embed:
            embeddings = self._batch_embed(chunks_to_embed)
            self._upsert_vectors_db(chunks_to_embed, embeddings, doc_id)

        # Step 6: 精确失效受影响块的缓存
        self._invalidate_affected_cache(doc_id, affected_chunks)

        # Step 7: 更新文档版本
        self._update_document_version(doc_id, new_hash)

        # Step 8: 发布更新事件
        self._publish_update_event(doc_id, affected_chunks)

        return {
            "status": "updated",
            "doc_id": doc_id,
            "affected_chunks": len(affected_chunks),
            "re_embedded_chunks": len(chunks_to_embed),
            "unchanged_chunks": len(unaffected_chunks),
            "old_hash": old_hash,
            "new_hash": new_hash,
        }

    def _detect_diff_regions(
        self, old_content: str, new_content: str
    ) -> List[DiffRegion]:
        """检测两个文本之间的差异区域"""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        regions = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            regions.append(DiffRegion(
                start_line=i1,
                end_line=max(i2 - 1, i1),
                operation={
                    "replace": "modify",
                    "delete": "delete",
                    "insert": "add",
                }.get(tag, tag),
            ))

        return regions

    def _identify_affected_chunks(
        self,
        diff_regions: List[DiffRegion],
        old_chunks: List[ChunkInfo],
    ) -> Tuple[List[ChunkInfo], List[ChunkInfo]]:
        """
        识别受变更影响的块

        策略：如果一个块的任一行在变更区域内，或离变更区域的边界
        在 CONTEXT_WINDOW_SIZE 个块以内，则认为受该块影响。
        """
        affected_set: Set[str] = set()

        for region in diff_regions:
            for chunk in old_chunks:
                # 检查块是否与变更区域重叠
                if (
                    chunk.start_line <= region.end_line
                    and chunk.end_line >= region.start_line
                ):
                    affected_set.add(chunk.chunk_id)

        # 扩展上下文窗口
        chunk_order = {c.chunk_id: i for i, c in enumerate(old_chunks)}
        extended_affected = set(affected_set)
        for chunk_id in affected_set:
            idx = chunk_order.get(chunk_id)
            if idx is None:
                continue
            # 包含前后窗口块
            for offset in range(
                -self.CONTEXT_WINDOW_SIZE,
                self.CONTEXT_WINDOW_SIZE + 1
            ):
                neighbor_idx = idx + offset
                if 0 <= neighbor_idx < len(old_chunks):
                    extended_affected.add(
                        old_chunks[neighbor_idx].chunk_id
                    )

        affected = [c for c in old_chunks if c.chunk_id in extended_affected]
        unaffected = [c for c in old_chunks if c.chunk_id not in extended_affected]
        return affected, unaffected

    def _rechunk_affected_regions(
        self,
        new_content: str,
        diff_regions: List[DiffRegion],
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[ChunkInfo]:
        """仅对受影响区域重新分块"""
        lines = new_content.splitlines(keepends=True)
        new_chunks = []

        for region in diff_regions:
            # 包含安全边界
            start = max(0, region.start_line - chunk_overlap)
            end = min(len(lines), region.end_line + chunk_overlap + 1)
            region_text = "".join(lines[start:end])

            # 重新分块
            chunks = self._text_splitter.split_text(
                region_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            for i, chunk_content in enumerate(chunks):
                new_chunks.append(ChunkInfo(
                    chunk_id=f"chunk_{region.start_line}_{i}",
                    content=chunk_content,
                    start_line=start,
                    end_line=end,
                ))

        return new_chunks

    def _batch_embed(self, chunks: List[ChunkInfo]) -> List[List[float]]:
        """批量生成向量"""
        texts = [c.content for c in chunks]
        return self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
        ).tolist()

    def _upsert_vectors_db(
        self,
        chunks: List[ChunkInfo],
        embeddings: List[List[float]],
        doc_id: str,
    ):
        """写入向量数据库（UPSERT 语义）"""
        records = []
        for chunk, embedding in zip(chunks, embeddings):
            records.append({
                "id": chunk.chunk_id,
                "vector": embedding,
                "doc_id": doc_id,
                "content": chunk.content[:1000],  # 截断存储
                "updated_at": "NOW()",
            })

        self.vector_db.upsert(
            collection_name="documents",
            data=records,
        )

    def _invalidate_affected_cache(
        self, doc_id: str, affected_chunks: List[ChunkInfo]
    ):
        """失效受影响块的缓存"""
        keys_to_delete = [f"doc:{doc_id}:*"]
        for chunk in affected_chunks:
            keys_to_delete.append(f"chunk:{chunk.chunk_id}")
            keys_to_delete.append(f"embedding:{chunk.chunk_id}")

        # 失效搜索结果缓存
        keys_to_delete.append(f"search:*")

        for key_pattern in keys_to_delete:
            if "*" in key_pattern:
                matched = self.cache.scan_keys(key_pattern)
                for key in matched:
                    self.cache.delete(key)
            else:
                self.cache.delete(key_pattern)

    def _update_document_version(self, doc_id: str, new_hash: str):
        """更新文档版本"""
        self.db.execute("""
            UPDATE documents
            SET content_hash = %s,
                version = version + 1,
                updated_at = NOW()
            WHERE id = %s
        """, (new_hash, doc_id))

    def _publish_update_event(
        self, doc_id: str, affected_chunks: List[ChunkInfo]
    ):
        """发布更新事件"""
        self.event_bus.publish(
            topic="knowledgebase.events",
            key=doc_id,
            value={
                "event_type": "DOC_UPDATED",
                "doc_id": doc_id,
                "affected_chunk_count": len(affected_chunks),
                "affected_chunk_ids": [c.chunk_id for c in affected_chunks],
                "timestamp": "NOW()",
            },
        )
```

### 11.1.5 增量更新触发机制

在实际系统中，增量更新的触发方式有多种：

```
触发机制选择决策树:

需要实时性？
├── 是 → 需要自动变更检测？
│        ├── 是 → 文件系统场景？
│        │        ├── 是 → inotify / WatchService / fsnotify
│        │        └── 否 → 数据库CDC (Debezium + Kafka)
│        └── 否 → Webhook / API 主动推送
└── 否 → 定时批量处理？
         ├── 低延迟容忍 → Cron Job (每5分钟)
         └── 高延迟容忍 → 日终批量 (ETL Pipeline)
```

```python
"""
变更检测器 —— 多种策略
"""
import hashlib
import os
from datetime import datetime
from typing import Optional


class ChangeDetector:
    """变更检测器，支持多种策略"""

    @staticmethod
    def by_file_hash(file_path: str) -> str:
        """策略1: 文件哈希检测（最可靠）"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()

    @staticmethod
    def by_timestamp(file_path: str) -> float:
        """策略2: 时间戳检测（最快，但不可靠）"""
        return os.path.getmtime(file_path)

    @staticmethod
    def by_size(file_path: str) -> int:
        """策略3: 文件大小检测（快速初筛）"""
        return os.path.getsize(file_path)

    @staticmethod
    def by_db_version(doc_id: str, db_client) -> int:
        """策略4: 数据库版本号检测"""
        row = db_client.find_one("documents", {"id": doc_id})
        return row.get("version", 0) if row else 0

    @classmethod
    def has_changed(
        cls,
        doc_id: str,
        file_path: Optional[str] = None,
        db_client=None,
        strategy: str = "hybrid",
    ) -> bool:
        """
        判断文档是否变更

        混合策略 (hybrid):
        1. 先用文件大小快速初筛
        2. 大小变化 → 肯定变更 → 返回 True
        3. 大小未变 → 计算哈希确认
        4. 哈希未变 → 检查数据库版本号
        """
        if strategy == "hash" and file_path:
            new_hash = cls.by_file_hash(file_path)
            old_hash = db_client.get_doc_hash(doc_id)
            return new_hash != old_hash

        elif strategy == "timestamp" and file_path:
            new_mtime = cls.by_timestamp(file_path)
            old_mtime = db_client.get_doc_mtime(doc_id)
            return new_mtime > old_mtime

        elif strategy == "hybrid":
            # 多层检测：速度快 → 慢，逐步确认
            if file_path:
                # 层1: 大小检查（毫秒级）
                new_size = cls.by_size(file_path)
                old_size = db_client.get_doc_size(doc_id)
                if new_size != old_size:
                    return True

                # 层2: 哈希检查（秒级）
                new_hash = cls.by_file_hash(file_path)
                old_hash = db_client.get_doc_hash(doc_id)
                if new_hash != old_hash:
                    return True

            # 层3: 版本检查（数据库查询）
            if db_client:
                new_version = cls.by_db_version(doc_id, db_client)
                old_version = db_client.get_doc_version(doc_id)
                return new_version > old_version

            return False

        raise ValueError(f"Unknown strategy: {strategy}")
```

### 11.1.6 工程案例：企业文档管理平台

某金融科技公司（管理50万+文档，日均更新2000次）的实践：

**问题**：初始方案是每次文档更新做全量重分块+全量重向量化，导致：
- 单次更新耗时15-30秒
- Embedding API 调用量暴增，月成本超预算300%
- 更新期间搜索结果短暂不完整（旧向量已删，新向量未就绪）

**解决方案**：
1. 引入增量更新机制 (如上文代码)
2. 90%的更新为小范围修改（单个段落），仅重新向量化1-3个块
3. 更新耗时降至0.5-2秒
4. Embedding API 月成本降低72%

---

## 11.2 一致性机制深度剖析

### 11.2.1 级联删除机制

#### 概念与设计目标

级联删除（Cascading Delete）确保删除一条文档记录时，所有关联数据一并清除，不留 "孤儿数据"。

```
                    文档表 (documents)
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
       块表(chunks)   向量DB(vectors)   索引(search_index)
            │
            ▼
        缓存(Redis)
```

#### 软删除模式（Tombstone Pattern）

```python
"""
软删除实现 —— Tombstone Pattern
"""
from datetime import datetime, timedelta
from enum import Enum


class DocStatus(Enum):
    ACTIVE = "active"
    DELETED = "deleted"       # 软删除，可恢复
    ARCHIVED = "archived"     # 归档
    PURGED = "purged"         # 永久清除


class SoftDeleteService:
    """
    软删除服务

    特点：
    - 数据不物理删除，仅标记状态
    - 设置恢复窗口期（默认30天）
    - 超过窗口期后，由清理任务物理删除
    - 支持一键恢复

    优势：
    - 用户误删可恢复，降低运维成本
    - 向量数据保留，恢复时无需重新向量化
    - 审计合规：所有操作留痕
    """

    RECOVERY_WINDOW_DAYS = 30

    def soft_delete(self, doc_id: str) -> Dict:
        """软删除文档"""
        now = datetime.utcnow()
        recovery_deadline = now + timedelta(days=self.RECOVERY_WINDOW_DAYS)

        # 标记文档
        self.db.update("documents", {"id": doc_id}, {
            "status": DocStatus.DELETED.value,
            "deleted_at": now.isoformat(),
            "deleted_by": get_current_user(),
            "recovery_deadline": recovery_deadline.isoformat(),
        })

        # 标记所有关联块
        self.db.update("chunks", {"doc_id": doc_id}, {
            "status": DocStatus.DELETED.value,
            "deleted_at": now.isoformat(),
        })

        # 向量数据库：标记而非删除（保留向量）
        self.vector_db.set_payload(
            filter_expr=f'doc_id == "{doc_id}"',
            payload={"status": "deleted", "deleted_at": now.isoformat()}
        )

        # 缓存：设置墓碑条目（告知缓存层该文档已删除）
        self.cache.set(
            f"tombstone:doc:{doc_id}",
            {"status": "deleted", "deleted_at": now.isoformat()},
            ttl=self.RECOVERY_WINDOW_DAYS * 86400,
        )

        # 搜索索引：标记为已删除（查询时过滤）
        self.search_index.update_by_query(
            index="documents",
            query={"term": {"doc_id": doc_id}},
            script="ctx._source.status = 'deleted'",
        )

        # 注册清理任务
        self._schedule_purge_task(doc_id, recovery_deadline)

        return {"status": "soft_deleted", "recovery_deadline": recovery_deadline.isoformat()}

    def recover_document(self, doc_id: str) -> Dict:
        """恢复软删除文档"""
        doc = self.db.find_one("documents", {"id": doc_id})

        if not doc or doc["status"] != DocStatus.DELETED.value:
            raise ValueError(f"Document {doc_id} not in deleted state")

        if datetime.utcnow() > datetime.fromisoformat(doc["recovery_deadline"]):
            raise ValueError(
                f"Recovery window expired for document {doc_id}"
            )

        # 恢复文档状态
        self.db.update("documents", {"id": doc_id}, {
            "status": DocStatus.ACTIVE.value,
            "deleted_at": None,
            "deleted_by": None,
            "recovery_deadline": None,
            "recovered_at": datetime.utcnow().isoformat(),
        })

        # 恢复所有块
        self.db.update("chunks", {"doc_id": doc_id}, {
            "status": DocStatus.ACTIVE.value,
            "deleted_at": None,
        })

        # 恢复向量标记
        self.vector_db.set_payload(
            filter_expr=f'doc_id == "{doc_id}"',
            payload={"status": "active"},
        )

        # 删除墓碑标记
        self.cache.delete(f"tombstone:doc:{doc_id}")

        # 恢复搜索索引
        self.search_index.update_by_query(
            index="documents",
            query={"term": {"doc_id": doc_id}},
            script="ctx._source.status = 'active'",
        )

        # 取消清理任务
        self._cancel_purge_task(doc_id)

        return {"status": "recovered"}

    def purge_expired_documents(self):
        """
        定时清理任务（由调度器定期执行）

        清理所有超过恢复窗口期的软删除文档
        """
        now = datetime.utcnow()
        expired_docs = self.db.find("documents", {
            "status": DocStatus.DELETED.value,
            "recovery_deadline": {"$lt": now.isoformat()},
        })

        for doc in expired_docs:
            doc_id = doc["id"]
            try:
                # 物理删除向量
                self.vector_db.delete_by_filter(
                    filter_expr=f'doc_id == "{doc_id}"'
                )
                # 物理删除块
                self.db.delete("chunks", {"doc_id": doc_id})
                # 删除搜索索引
                self.search_index.delete_by_query(
                    index="documents",
                    query={"term": {"doc_id": doc_id}},
                )
                # 标记文档为已清除
                self.db.update("documents", {"id": doc_id}, {
                    "status": DocStatus.PURGED.value,
                    "purged_at": now.isoformat(),
                })
                logger.info(f"Purged document {doc_id}")
            except Exception as e:
                logger.error(f"Failed to purge document {doc_id}: {e}")
```

### 11.2.2 版本管理机制

```
版本管理核心数据结构:

┌──────────────────────────────────────────────┐
│              文档版本树                        │
│                                              │
│   v1.0 ──▶ v1.1 ──▶ v2.0 ──▶ v2.1 (current) │
│              │                    │           │
│              └──▶ v1.1-hotfix    (分支)       │
│                                              │
│  每层映射关系:                                │
│  doc_version_id ──▶ chunk_version_ids         │
│                  ──▶ vector_version_ids       │
│                  ──▶ index_version_id         │
└──────────────────────────────────────────────┘
```

#### 数据库Schema：版本管理

```sql
-- 文档版本表
CREATE TABLE document_versions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id          UUID NOT NULL REFERENCES documents(id),
    version_number  INTEGER NOT NULL,
    content_hash    VARCHAR(64) NOT NULL,
    content_snapshot TEXT,                    -- 或存储在对象存储
    change_summary  TEXT,                     -- 变更摘要
    chunk_version_ids UUID[],                -- 关联的块版本ID列表
    vector_version_id UUID,                  -- 关联的向量版本ID
    index_version_id UUID,                   -- 关联的索引版本ID
    created_by      VARCHAR(128),
    created_at      TIMESTAMP DEFAULT NOW(),
    is_active       BOOLEAN DEFAULT TRUE,
    UNIQUE (doc_id, version_number)
);

-- 块版本表
CREATE TABLE chunk_versions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id        UUID NOT NULL,
    doc_version_id  UUID NOT NULL REFERENCES document_versions(id),
    content         TEXT NOT NULL,
    content_hash    VARCHAR(64) NOT NULL,
    embedding_hash  VARCHAR(64),             -- 向量哈希，用于判断是否需要重新向量化
    start_position  INTEGER,
    end_position    INTEGER,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- 向量版本表
CREATE TABLE vector_versions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_version_id  UUID NOT NULL REFERENCES document_versions(id),
    chunk_version_id UUID NOT NULL REFERENCES chunk_versions(id),
    vector_id       VARCHAR(256) NOT NULL,  -- 向量数据库中的ID
    model_name      VARCHAR(128) NOT NULL,  -- 使用的Embedding模型
    model_version   VARCHAR(64),            -- 模型版本
    dimensions      INTEGER NOT NULL,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- 索引
CREATE INDEX idx_doc_versions_doc_id ON document_versions(doc_id);
CREATE INDEX idx_doc_versions_active ON document_versions(doc_id) WHERE is_active = TRUE;
CREATE INDEX idx_chunk_versions_doc_ver ON chunk_versions(doc_version_id);
```

#### 版本回滚实现

```python
"""
文档版本回滚服务
"""
class VersionRollbackService:
    """
    版本回滚服务

    回滚策略：
    1. 加载目标版本的快照
    2. 激活目标版本的块和向量
    3. 停用当前版本
    4. 更新缓存和索引
    """

    def rollback_to_version(self, doc_id: str, target_version: int) -> Dict:
        """回滚文档到指定版本"""
        # 1. 验证目标版本存在
        target = self.db.find_one("document_versions", {
            "doc_id": doc_id,
            "version_number": target_version,
        })
        if not target:
            raise VersionNotFoundError(
                f"Version {target_version} not found for doc {doc_id}"
            )

        # 2. 加载目标版本的块
        target_chunks = self.db.find("chunk_versions", {
            "doc_version_id": target["id"],
        })

        # 3. 加载目标版本的向量
        target_vectors = self.db.find("vector_versions", {
            "doc_version_id": target["id"],
        })

        # 4. 原子切换
        with self.db.transaction():
            # 停用当前版本
            self.db.update(
                "document_versions",
                {"doc_id": doc_id, "is_active": True},
                {"is_active": False}
            )

            # 激活目标版本
            self.db.update(
                "document_versions",
                {"id": target["id"]},
                {"is_active": True}
            )

            # 恢复文档内容
            self.db.update("documents", {"id": doc_id}, {
                "content": target["content_snapshot"],
                "version": target_version,
                "updated_at": "NOW()",
            })

            # 恢复块
            chunk_mapping = {}  # old_chunk_id -> new_chunk_id
            for cv in target_chunks:
                new_chunk_id = str(uuid.uuid4())
                chunk_mapping[cv["chunk_id"]] = new_chunk_id
                self.db.insert("chunks", {
                    "id": new_chunk_id,
                    "doc_id": doc_id,
                    "content": cv["content"],
                    "version": target_version,
                })

            # 重新激活向量（如向量DB支持版本切换，直接切换）
            vector_ids = [v["vector_id"] for v in target_vectors]

            # 方案A：如果向量DB支持版本化查询
            self.vector_db.set_active_version(
                doc_id=doc_id,
                version=target_version,
            )

        # 5. 失效缓存
        self.cache.delete_pattern(f"doc:{doc_id}:*")
        self.cache.delete_pattern(f"search:*")

        # 6. 更新搜索索引
        self.search_index.reindex_document(doc_id, target["content_snapshot"])

        return {
            "status": "rolled_back",
            "doc_id": doc_id,
            "from_version": self._get_current_version(doc_id),
            "to_version": target_version,
        }
```

### 11.2.3 增量向量化(Embedding)策略

对于大型文档库，全量重新向量化代价极高。增量向量化是性能优化的关键。

```
增量向量化决策流程:

文档更新
    │
    ▼
内容哈希变了？
    ├── 否 → 跳过向量化（内容未变）
    │
    └── 是 → 块内容变了？
              ├── 否 → 保留现有向量（仅元数据变了）
              │
              └── 是 → 仅对变更块重新向量化
                        │
                        ▼
                   向量模型版本变了？
                        ├── 否 → 仅变更块
                        │
                        └── 是 → 需要全量重建
                                (模型升级场景)
```

```python
"""
增量向量化引擎
"""
class IncrementalEmbeddingEngine:
    """
    增量向量化引擎

    核心优化：
    1. 块级别哈希比对，只向量化真正变化的块
    2. 批量处理：攒批后统一调用 Embedding API
    3. 优先级队列：热点文档优先更新
    """

    def __init__(
        self,
        embedding_model,
        vector_db,
        db_client,
        batch_size: int = 32,
        max_concurrent_batches: int = 4,
    ):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.db = db_client
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches

    def incremental_embed(
        self,
        doc_id: str,
        new_chunks: List[ChunkInfo],
        existing_chunks: List[ChunkInfo],
    ) -> Dict:
        """
        增量向量化：仅处理变化了的块
        """
        # 建立现有块映射 (content_hash -> embedding_id)
        existing_map = {
            self._content_hash(c.content): c
            for c in existing_chunks
        }

        to_embed = []
        to_reuse = []

        for chunk in new_chunks:
            chunk_hash = self._content_hash(chunk.content)
            if chunk_hash in existing_map:
                # 内容未变，复用现有向量
                to_reuse.append({
                    "chunk": chunk,
                    "existing_embedding_id": existing_map[chunk_hash].chunk_id,
                })
            else:
                to_embed.append(chunk)

        logger.info(
            f"Doc {doc_id}: {len(to_embed)} chunks to embed, "
            f"{len(to_reuse)} chunks to reuse"
        )

        # 批量向量化新块
        embeddings_map = {}
        for i in range(0, len(to_embed), self.batch_size):
            batch = to_embed[i:i + self.batch_size]
            texts = [c.content for c in batch]
            batch_embeddings = self.embedding_model.encode(texts)
            for chunk, emb in zip(batch, batch_embeddings):
                embeddings_map[chunk.chunk_id] = emb.tolist()

        # 写入向量数据库
        self._upsert_embeddings(doc_id, embeddings_map, to_reuse)

        return {
            "total_chunks": len(new_chunks),
            "embedded": len(to_embed),
            "reused": len(to_reuse),
            "savings_pct": round(len(to_reuse) / max(len(new_chunks), 1) * 100, 1),
        }

    @staticmethod
    def _content_hash(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()
```

### 11.2.4 索引重建策略

```
                        索引重建方式对比

┌──────────────┬─────────────────┬──────────────────┬─────────────────┐
│   维度       │  全量重建       │  滚动重建        │  零停机重建     │
│              │  Full Rebuild   │  Rolling Rebuild │  Zero-downtime  │
├──────────────┼─────────────────┼──────────────────┼─────────────────┤
│ 原理         │ 删除旧索引，    │ 逐段替换旧索引   │ 新索引并行构建  │
│              │ 重新构建全量    │                  │ 完成后切换别名  │
├──────────────┼─────────────────┼──────────────────┼─────────────────┤
│ 停机时间     │ 有（分钟-小时） │ 无               │ 无              │
├──────────────┼─────────────────┼──────────────────┼─────────────────┤
│ 实现复杂度   │ 低              │ 中               │ 高              │
├──────────────┼─────────────────┼──────────────────┼─────────────────┤
│ 资源消耗     │ 高（峰值）      │ 中（均匀分布）   │ 高（双倍资源）  │
├──────────────┼─────────────────┼──────────────────┼─────────────────┤
│ 一致性保证   │ 强一致          │ 最终一致         │ 强一致          │
├──────────────┼─────────────────┼──────────────────┼─────────────────┤
│ 回滚能力     │ 困难            │ 容易（逐段回滚） │ 容易（别名切换）│
├──────────────┼─────────────────┼──────────────────┼─────────────────┤
│ 适用场景     │ 模型升级        │ 大索引定期维护   │ 关键业务索引    │
│              │ 索引损坏恢复    │ 索引参数调整     │ 高频查询场景    │
└──────────────┴─────────────────┴──────────────────┴─────────────────┘
```

#### 零停机索引重建实现

```python
"""
零停机索引重建器 — Blue/Green 索引切换
"""
class ZeroDowntimeRebuilder:
    """
    使用 Blue/Green 策略实现零停机索引重建

    流程:
    1. 创建新索引 (green_index)
    2. 全量数据写入新索引
    3. 验证新索引完整性
    4. 原子切换别名 (alias: current -> green_index)
    5. 删除旧索引 (blue_index)
    """

    ALIAS_NAME = "documents_current"
    INDEX_PREFIX = "documents_v"

    def rebuild(self, search_client) -> Dict:
        # 1. 确定新索引名
        current_index = search_client.get_alias_target(self.ALIAS_NAME)
        version = int(current_index.split("_v")[-1]) if current_index else 0
        new_index = f"{self.INDEX_PREFIX}{version + 1}"

        logger.info(f"Starting zero-downtime rebuild: {current_index} -> {new_index}")

        # 2. 创建新索引（优化配置）
        search_client.create_index(
            index=new_index,
            settings={
                "number_of_shards": 3,
                "number_of_replicas": 1,
                "refresh_interval": "-1",  # 构建期间暂停刷新
            },
            mappings=self._get_optimized_mappings(),
        )

        # 3. 批量导入数据
        try:
            self._bulk_import(search_client, new_index)
        except Exception as e:
            search_client.delete_index(new_index)
            raise IndexRebuildError(f"Bulk import failed: {e}")

        # 4. 恢复刷新并强制合并
        search_client.update_settings(new_index, {
            "refresh_interval": "30s",
        })
        search_client.force_merge(
            index=new_index,
            max_num_segments=5,
        )

        # 5. 验证新索引
        doc_count_new = search_client.count(index=new_index)
        doc_count_current = search_client.count(index=current_index) if current_index else 0

        if doc_count_new < doc_count_current * 0.99:
            search_client.delete_index(new_index)
            raise IndexRebuildError(
                f"Validation failed: new={doc_count_new} < expected={doc_count_current}"
            )

        # 6. 原子切换别名
        actions = [
            {"remove": {"index": current_index, "alias": self.ALIAS_NAME}},
            {"add": {"index": new_index, "alias": self.ALIAS_NAME}},
        ] if current_index else [
            {"add": {"index": new_index, "alias": self.ALIAS_NAME}},
        ]
        search_client.update_aliases(actions)

        # 7. 删除旧索引（延迟删除，便于回滚）
        if current_index:
            # 保留旧索引24小时作为回滚备份
            self._schedule_index_cleanup(current_index, delay_hours=24)

        logger.info(f"Rebuild complete: {new_index} is now active")
        return {
            "status": "completed",
            "old_index": current_index,
            "new_index": new_index,
            "doc_count": doc_count_new,
        }
```

### 11.2.5 同步策略对比

```
                    同步策略综合对比

┌────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│    策略        │  强一致性    │  最终一致性  │  最终一致性  │  多级缓存    │
│                │  (同步)      │  (异步消息)  │  (定时轮询)  │  (写穿+失效) │
├────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 延迟           │ 0-50ms       │ 10-500ms     │ 1-60s        │ 50-200ms     │
├────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 吞吐量         │ 低           │ 高           │ 中           │ 高           │
├────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 复杂性         │ 低           │ 中           │ 低           │ 高           │
├────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 故障恢复       │ 困难         │ 自动         │ 自动         │ 半自动       │
├────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 数据丢失风险   │ 无           │ 极低         │ 有（窗口内） │ 低           │
├────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ 适用场景       │ 金融/合规    │ 一般企业     │ 低实时性     │ 高并发查询   │
└────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

---

## 11.3 防腐蚀机制

> **腐蚀**（Corruption）概念：指随着时间推移和操作累积，知识库数据质量逐步退化的现象。包括脏数据、孤儿向量、过期索引等。

### 11.3.1 防止脏数据：验证管线

```
数据验证管线 (Validation Pipeline)

原始文档
    │
    ▼
┌─────────────────┐
│ Layer 1: 格式验证│  文件类型、大小、编码检查
│ - 白名单: .pdf, │  拒绝: .exe, .zip, 空文件, >100MB
│   .docx, .txt   │
│ - 编码: UTF-8   │
│ - 大小: <100MB  │
└────────┬────────┘
         │ 通过
         ▼
┌─────────────────┐
│ Layer 2: 内容验证│  内容质量检查
│ - 可读文本比例  │  拒绝: 纯图片PDF、乱码、空内容
│   > 30%         │
│ - 语言检测      │
│ - 敏感词过滤    │
└────────┬────────┘
         │ 通过
         ▼
┌─────────────────┐
│ Layer 3: Schema  │  字段完整性
│ 强制验证         │
│ - 必填字段检查  │  拒绝: 缺少标题、无doc_id
│ - 类型检查      │
│ - 唯一性检查    │
└────────┬────────┘
         │ 通过
         ▼
┌─────────────────┐
│ Layer 4: 业务规则│  业务逻辑验证
│ - 去重检查      │  警告: 与已有文档相似度>95%
│   (相似度<95%)  │  转为人工审核
│ - 分区权限      │
│ - 合规标签      │
└────────┬────────┘
         │ 通过
         ▼
     进入处理流程
```

```python
"""
多层数据验证管线
"""
from dataclasses import dataclass, field
from typing import List, Optional, Callable


@dataclass
class ValidationResult:
    """单层验证结果"""
    layer: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ValidationPipeline:
    """多层验证管线"""

    def __init__(self):
        self.layers: List[Callable] = []

    def add_layer(self, name: str, validator: Callable):
        self.layers.append((name, validator))

    def validate(self, document: Dict) -> List[ValidationResult]:
        results = []
        for layer_name, validator in self.layers:
            result = validator(document)
            result.layer = layer_name
            results.append(result)
            if not result.passed:
                # 快速失败：任一层失败则停止
                break
        return results


def format_validator(doc: Dict) -> ValidationResult:
    """格式验证层"""
    result = ValidationResult(layer="format", passed=True)
    allowed_types = {"pdf", "docx", "txt", "md", "html"}
    if doc.get("file_type") not in allowed_types:
        result.passed = False
        result.errors.append(f"Unsupported type: {doc['file_type']}")
    if doc.get("file_size", 0) > 100 * 1024 * 1024:
        result.passed = False
        result.errors.append("File too large (>100MB)")
    return result


def content_validator(doc: Dict) -> ValidationResult:
    """内容验证层"""
    result = ValidationResult(layer="content", passed=True)
    content = doc.get("content", "")
    if not content or len(content.strip()) < 50:
        result.passed = False
        result.errors.append("Content too short or empty")
    # 检查可读文本比例
    total_chars = len(content)
    if total_chars > 0:
        printable = sum(1 for c in content if c.isprintable() or c in '\n\r\t')
        if printable / total_chars < 0.3:
            result.passed = False
            result.errors.append("Too many non-printable characters")
    return result


def dedup_validator(doc: Dict) -> ValidationResult:
    """去重验证层"""
    result = ValidationResult(layer="dedup", passed=True)
    content_hash = hashlib.sha256(doc["content"].encode()).hexdigest()
    # 检查数据库中是否存在相同哈希
    existing = db.find_one("documents", {"content_hash": content_hash})
    if existing:
        result.passed = False
        result.errors.append(f"Duplicate document: existing ID={existing['id']}")
    return result


# 组装管道
pipeline = ValidationPipeline()
pipeline.add_layer("format", format_validator)
pipeline.add_layer("content", content_validator)
pipeline.add_layer("dedup", dedup_validator)
```

### 11.3.2 防止孤儿向量

孤儿向量（Orphan Vectors）指的是向量数据库中那些没有对应文档或块记录的向量条目。这些向量会导致"幽灵检索"——检索到不存在的内容。

```python
"""
孤儿向量检测与清理
"""
class OrphanVectorDetector:
    """
    孤儿向量检测器

    检测策略：
    1. 定期全量扫描：对比向量DB和关系型DB的 chunk_id
    2. 外键追踪：在向量条目中记录 doc_id，定期验证
    3. 变更事件触发：删除文档后异步验证向量已清除
    """

    def __init__(self, vector_db, relational_db):
        self.vector_db = vector_db
        self.relational_db = relational_db

    def full_scan(self, batch_size: int = 10000) -> Dict:
        """
        全量扫描孤儿向量

        返回孤儿向量列表和统计信息
        """
        # 从关系数据库获取所有活跃chunk_id
        active_chunk_ids = set()
        offset = 0
        while True:
            batch = self.relational_db.find(
                "chunks",
                {"status": "active"},
                fields=["id"],
                limit=batch_size,
                offset=offset,
            )
            if not batch:
                break
            for row in batch:
                active_chunk_ids.add(row["id"])
            offset += batch_size

        # 从向量数据库获取所有chunk_id
        vector_ids = set()
        offset = 0
        while True:
            batch = self.vector_db.scroll(
                collection="documents",
                fields=["doc_id", "chunk_id"],
                limit=batch_size,
                offset=offset,
            )
            if not batch:
                break
            for item in batch:
                vector_ids.add(item["chunk_id"])
            offset += batch_size

        # 找出孤儿向量：在向量DB但不在关系DB中的
        orphans = vector_ids - active_chunk_ids
        # 找出缺失向量：在关系DB但不在向量DB中的
        missing = active_chunk_ids - vector_ids

        return {
            "total_vectors": len(vector_ids),
            "total_chunks": len(active_chunk_ids),
            "orphan_vectors": list(orphans),
            "orphan_count": len(orphans),
            "missing_vectors": list(missing),
            "missing_count": len(missing),
        }

    def clean_orphans(self, dry_run: bool = True) -> Dict:
        """清理孤儿向量"""
        scan_result = self.full_scan()
        orphans = scan_result["orphan_vectors"]

        if not dry_run and orphans:
            # 分批删除
            for i in range(0, len(orphans), 1000):
                batch = orphans[i:i + 1000]
                self.vector_db.delete_by_ids(batch)

        return {
            "dry_run": dry_run,
            "orphans_found": len(orphans),
            "orphans_deleted": len(orphans) if not dry_run else 0,
        }

    def setup_periodic_scan(self, interval_hours: int = 6):
        """设置定期扫描任务"""
        scheduler.add_job(
            self.clean_orphans,
            trigger="interval",
            hours=interval_hours,
            kwargs={"dry_run": False},
            id="orphan_vector_scan",
            name="Orphan Vector Scanner",
        )
```

#### 孤儿向量产生原因与预防

```
孤儿向量产生路径及预防措施：

原因1: 删除操作部分失败
├── 症状: 文档已删，向量未删
├── 预防: 事务包装 + 补偿任务
└── 检测: periodic_scan

原因2: 并发更新导致的竞态
├── 症状: 旧块ID被替换，但旧向量仍在
├── 预防: 乐观锁(version) + 幂等写入
└── 检测: chunk-vector 一致性校验

原因3: 向量DB写入失败
├── 症状: 块已创建，向量写入失败
├── 预防: Write-Ahead Log + 重试队列
└── 检测: missing_vectors 扫描

原因4: 手动数据库操作
├── 症状: DBA直接删除了关系DB记录
├── 预防: 权限管控 + 操作审计
└── 检测: foreign_key_tracking
```

### 11.3.3 防止过期索引

```
过期索引的生命周期管理:

┌─────────────────────────────────────────────────┐
│             过期索引检测策略                      │
│                                                 │
│  策略1: 事件驱动失效                            │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ 文档更新  │───▶│ 发布事件  │───▶│ 索引更新  │  │
│  └──────────┘    └──────────┘    └──────────┘  │
│  延迟: <100ms                                   │
│                                                 │
│  策略2: TTL自动过期                             │
│  ┌──────────┐         ┌──────────┐             │
│  │ 写入索引  │────────▶│ TTL=3600s│             │
│  │ 设置TTL   │         │ 自动清理  │             │
│  └──────────┘         └──────────┘             │
│  适用: 实时性要求低的缓存索引                     │
│                                                 │
│  策略3: 版本号比对                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ 索引条目  │    │ 对比版本  │    │ 过期则更新│  │
│  │ version=5│ vs │ 最新ver=6│───▶│ 或删除    │  │
│  └──────────┘    └──────────┘    └──────────┘  │
│  适用: Elasticsearch等支持字段级版本控制的引擎    │
└─────────────────────────────────────────────────┘
```

### 11.3.4 数据完整性校验

```python
"""
数据完整性校验服务
"""
class IntegrityChecker:
    """
    多层数据完整性校验

    校验内容：
    1. 计数一致性: COUNT(文档) vs COUNT(块) vs COUNT(向量)
    2. 内容一致性: 文档哈希 vs 块内容哈希
    3. 向量一致性: 向量条目数与预期匹配
    """

    def run_all_checks(self) -> Dict[str, Dict]:
        """运行全部完整性检查"""
        results = {}
        results["count_check"] = self.check_counts()
        results["content_check"] = self.check_content_hashes()
        results["vector_check"] = self.check_vector_consistency()
        results["cache_check"] = self.check_cache_consistency()

        all_pass = all(
            r.get("status") == "pass" for r in results.values()
        )
        return {
            "overall_status": "pass" if all_pass else "fail",
            "checks": results,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def check_counts(self) -> Dict:
        """计数一致性检查"""
        doc_count = self.db.count("documents", {"status": "active"})
        chunk_count = self.db.count("chunks", {"status": "active"})
        vector_count = self.vector_db.count(collection="documents")

        # 块数应 >= 文档数
        # 向量数应 == 块数
        issues = []
        if chunk_count < doc_count:
            issues.append(f"chunk_count({chunk_count}) < doc_count({doc_count})")
        if vector_count != chunk_count:
            issues.append(
                f"vector_count({vector_count}) != chunk_count({chunk_count})"
            )

        return {
            "status": "pass" if not issues else "fail",
            "doc_count": doc_count,
            "chunk_count": chunk_count,
            "vector_count": vector_count,
            "issues": issues,
        }

    def check_content_hashes(self) -> Dict:
        """
        内容-向量一致性检查

        采样策略：随机抽取N个块，重新计算向量并与数据库中的比对
        """
        sample_size = 100
        chunks = self.db.find_random("chunks", sample_size)
        mismatches = []

        for chunk in chunks:
            # 重新生成向量
            new_embedding = self.embedding_model.encode([chunk["content"]])[0]
            # 从向量DB获取现有向量
            existing = self.vector_db.get(chunk["id"])
            if existing:
                # 计算余弦相似度
                similarity = cosine_similarity(new_embedding, existing["vector"])
                if similarity < 0.99:  # 允许1%误差
                    mismatches.append({
                        "chunk_id": chunk["id"],
                        "similarity": similarity,
                    })

        return {
            "status": "pass" if not mismatches else "fail",
            "samples_checked": sample_size,
            "mismatches_found": len(mismatches),
            "sample_mismatches": mismatches[:5],
        }
```

---

## 11.4 完整同步架构

### 11.4.1 总体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        知识库完整同步架构 (Event-Driven)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐          │
│  │ 文档导入  │     │ API 网关  │     │ Webhook  │     │ 定时任务  │          │
│  │ (Upload) │     │ (REST)   │     │ (Push)   │     │ (Cron)   │          │
│  └────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘          │
│       │                │                │                │                 │
│       └────────────────┴────────┬───────┴────────────────┘                 │
│                                 │                                          │
│                                 ▼                                          │
│                   ┌─────────────────────────┐                              │
│                   │   写入协调器 (Write       │                              │
│                   │   Coordinator)            │                              │
│                   │   - 事务管理              │                              │
│                   │   - 版本分配              │                              │
│                   │   - 冲突检测              │                              │
│                   └────────────┬────────────┘                              │
│                                │                                           │
│              ┌─────────────────┼─────────────────┐                         │
│              ▼                 ▼                 ▼                          │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                    │
│   │ 主存储       │  │ 变更事件     │  │ 操作日志     │                    │
│   │ (PostgreSQL) │  │ (Kafka)      │  │ (WAL)        │                    │
│   │              │  │              │  │              │                    │
│   │ - documents  │  │ topic:       │  │ - 操作记录   │                    │
│   │ - chunks     │  │ kb.events    │  │ - 版本历史   │                    │
│   │ - versions   │  │              │  │ - 补偿任务   │                    │
│   └──────┬───────┘  └──────┬───────┘  └──────────────┘                    │
│          │                 │                                               │
│          │                 ▼                                               │
│          │    ┌────────────────────────┐                                   │
│          │    │  同步消费者组           │                                   │
│          │    │  (Sync Consumers)      │                                   │
│          │    │                        │                                   │
│          │    │  ┌──────────────────┐  │                                   │
│          │    │  │ Vector Syncer    │──┼────▶ 向量DB (Milvus/Qdrant)      │
│          │    │  └──────────────────┘  │                                   │
│          │    │  ┌──────────────────┐  │                                   │
│          │    │  │ Index Syncer     │──┼────▶ 搜索引擎 (Elasticsearch)     │
│          │    │  └──────────────────┘  │                                   │
│          │    │  ┌──────────────────┐  │                                   │
│          │    │  │ Cache Invalidation│──┼────▶ 缓存 (Redis)                │
│          │    │  └──────────────────┘  │                                   │
│          │    └────────────────────────┘                                   │
│          │                                                                 │
│          ▼                                                                 │
│  ┌──────────────────────────────────────────────┐                          │
│  │           监控与告警层                         │                          │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐  │                          │
│  │  │ 同步延迟  │ │ 一致性   │ │ 错误率/重试  │  │                          │
│  │  │ Metrics  │ │ Dashboard│ │ Alerts       │  │                          │
│  │  └──────────┘ └──────────┘ └──────────────┘  │                          │
│  └──────────────────────────────────────────────┘                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 11.4.2 事件驱动同步详细设计

```python
"""
基于 Kafka 的变更数据捕获 (CDC) 同步
"""
import json
from typing import Dict, Any
from confluent_kafka import Producer, Consumer, KafkaError


class KnowledgeBaseEventProducer:
    """
    知识库事件生产者

    所有文档变更都通过此生产者发布到 Kafka
    """

    TOPIC = "knowledgebase.events"

    EVENT_TYPES = {
        "DOC_CREATED": "doc.created",
        "DOC_UPDATED": "doc.updated",
        "DOC_DELETED": "doc.deleted",
        "DOC_VERSION_ROLLBACK": "doc.version.rollback",
        "CHUNK_CREATED": "chunk.created",
        "CHUNK_UPDATED": "chunk.updated",
        "CHUNK_DELETED": "chunk.deleted",
        "INDEX_REBUILT": "index.rebuilt",
    }

    def __init__(self, bootstrap_servers: str):
        self.producer = Producer({
            "bootstrap.servers": bootstrap_servers,
            "acks": "all",                    # 等待所有副本确认
            "enable.idempotence": True,       # 幂等生产者
            "compression.type": "snappy",     # 压缩
            "linger.ms": 5,                   # 微批处理
        })

    def publish_change(
        self,
        event_type: str,
        doc_id: str,
        payload: Dict[str, Any],
        version: int,
    ):
        """发布变更事件"""
        event = {
            "event_type": event_type,
            "doc_id": doc_id,
            "version": version,
            "timestamp": datetime.utcnow().isoformat(),
            "payload": payload,
        }

        # 使用 doc_id 作为分区键，保证同一文档的事件有序
        self.producer.produce(
            topic=self.TOPIC,
            key=doc_id.encode("utf-8"),
            value=json.dumps(event).encode("utf-8"),
            headers={
                "event_type": event_type.encode("utf-8"),
                "version": str(version).encode("utf-8"),
            },
            callback=self._delivery_callback,
        )
        self.producer.flush(timeout=5)

    @staticmethod
    def _delivery_callback(err, msg):
        if err:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(
                f"Message delivered to {msg.topic()} "
                f"[{msg.partition()}] offset={msg.offset()}"
            )


class VectorSyncConsumer:
    """
    向量数据库同步消费者

    消费 KB 变更事件，同步到向量数据库
    """

    def __init__(
        self,
        bootstrap_servers: str,
        vector_db,
        embedding_model,
        group_id: str = "vector-sync-group",
    ):
        self.consumer = Consumer({
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,     # 手动提交，保证至少一次处理
            "max.poll.interval.ms": 300000,  # 5分钟处理时间
        })
        self.consumer.subscribe(["knowledgebase.events"])
        self.vector_db = vector_db
        self.embedding_model = embedding_model

    def run(self):
        """持续运行同步循环"""
        while True:
            msg = self.consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error(f"Consumer error: {msg.error()}")
                continue

            try:
                event = json.loads(msg.value().decode("utf-8"))
                self._process_event(event)
                self.consumer.commit(msg)  # 处理成功后提交
            except Exception as e:
                logger.error(
                    f"Failed to process event offset={msg.offset()}: {e}"
                )
                # 不提交，等待重试
                # 死信队列处理逻辑见下文

    def _process_event(self, event: Dict):
        """处理单个事件"""
        event_type = event["event_type"]
        doc_id = event["doc_id"]
        payload = event["payload"]

        if event_type == "doc.created" or event_type == "doc.updated":
            chunks = payload.get("chunks", [])
            embeddings = self.embedding_model.encode(
                [c["content"] for c in chunks]
            )
            records = [
                {
                    "id": c["chunk_id"],
                    "vector": emb.tolist(),
                    "doc_id": doc_id,
                    "content": c["content"][:1000],
                }
                for c, emb in zip(chunks, embeddings)
            ]
            self.vector_db.upsert("documents", records)

        elif event_type == "doc.deleted":
            self.vector_db.delete_by_filter(
                filter_expr=f'doc_id == "{doc_id}"'
            )

        elif event_type == "doc.version.rollback":
            target_version = payload["target_version"]
            self.vector_db.set_active_version(doc_id, target_version)


class CacheInvalidationConsumer:
    """缓存失效消费者"""

    def __init__(self, bootstrap_servers, cache_client):
        self.consumer = Consumer({
            "bootstrap.servers": bootstrap_servers,
            "group.id": "cache-invalidation-group",
            "auto.offset.reset": "latest",     # 只关注新事件
        })
        self.consumer.subscribe(["knowledgebase.events"])
        self.cache = cache_client

    def _process_event(self, event: Dict):
        doc_id = event["doc_id"]
        event_type = event["event_type"]

        if event_type in ("doc.updated", "doc.deleted", "doc.version.rollback"):
            # 批量失效
            patterns = [
                f"doc:{doc_id}:*",
                f"search:*",
            ]
            if "affected_chunks" in event.get("payload", {}):
                for chunk_id in event["payload"]["affected_chunks"]:
                    patterns.append(f"chunk:{chunk_id}")
                    patterns.append(f"embedding:{chunk_id}")

            for pattern in patterns:
                keys = self.cache.scan_keys(pattern)
                if keys:
                    self.cache.delete(*keys)
```

### 11.4.3 事务设计：跨数据库原子性

```
跨 PostgreSQL(Master) + 向量DB 的事务设计:

┌─────────────────────────────────────────────────────┐
│              Saga 模式 事务协调                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Begin Transaction                                  │
│      │                                              │
│      ▼                                              │
│  ┌────────────────────┐                            │
│  │ Step 1: PG Write   │  写主表、块表               │
│  │ (本地事务)          │  成功 →                     │
│  └────────┬───────────┘                            │
│           │ 成功                                     │
│           ▼                                         │
│  ┌────────────────────┐                            │
│  │ Step 2: Vector     │  UPSERT 向量                │
│  │ DB Write           │  成功 →                      │
│  └────────┬───────────┘                            │
│           │ 失败                                     │
│           ▼                                         │
│  ┌────────────────────┐    ┌────────────────────┐  │
│  │ Compensating       │◀───│ Step 1 补偿:        │  │
│  │ Transaction:       │    │ 标记 PG记录为       │  │
│  │ 标记向量为待同步    │    │ "vector_pending"    │  │
│  └────────────────────┘    └────────────────────┘  │
│                                                     │
│  最终一致性保证:                                      │
│  - PG是真相之源 (Source of Truth)                     │
│  - 向量DB通过消费者异步追上                            │
│  - "vector_pending" 标记触发补偿任务                  │
└─────────────────────────────────────────────────────┘
```

```python
"""
Saga事务协调器
"""
class SagaCoordinator:
    """
    使用 Saga 模式协调跨存储写入

    当向量DB写入失败时：
    1. PG中的记录标记为 pending_sync
    2. 后台补偿任务轮询 pending_sync 记录并重试
    3. 超过最大重试次数后告警
    """

    MAX_RETRIES = 5

    def write_with_saga(self, doc_id: str, chunks: List[Dict]) -> Dict:
        """Saga写入"""
        saga_id = str(uuid.uuid4())

        try:
            # Step 1: PostgreSQL 写入（本地事务）
            with self.pg.transaction():
                self.pg.insert("documents", {"id": doc_id, "status": "syncing"})
                for chunk in chunks:
                    self.pg.insert("chunks", {
                        "id": chunk["id"],
                        "doc_id": doc_id,
                        "content": chunk["content"],
                        "sync_status": "pending",
                    })

                # 记录 Saga
                self.pg.insert("saga_log", {
                    "saga_id": saga_id,
                    "doc_id": doc_id,
                    "step": "pg_write",
                    "status": "completed",
                })

            # Step 2: 向量DB写入
            try:
                embeddings = self.embedding_model.encode(
                    [c["content"] for c in chunks]
                )
                self.vector_db.upsert("documents", [
                    {
                        "id": chunk["id"],
                        "vector": emb.tolist(),
                        "doc_id": doc_id,
                    }
                    for chunk, emb in zip(chunks, embeddings)
                ])

                # Step 2 成功：更新状态
                self.pg.update("chunks", {"doc_id": doc_id}, {
                    "sync_status": "synced"
                })
                self.pg.update("documents", {"id": doc_id}, {
                    "status": "active"
                })

                return {"saga_id": saga_id, "status": "completed"}

            except Exception as vec_error:
                logger.error(f"Vector DB write failed: {vec_error}")
                # PG标记为待同步，补偿任务稍后重试
                self.pg.update("documents", {"id": doc_id}, {
                    "status": "vector_pending",
                    "last_error": str(vec_error)[:500],
                })
                return {"saga_id": saga_id, "status": "partial", "error": str(vec_error)}

        except Exception as e:
            logger.error(f"Saga failed at PG step: {e}")
            return {"saga_id": saga_id, "status": "failed", "error": str(e)}


class CompensationWorker:
    """
    补偿工作器

    定期扫描 pending 状态的记录并重试向量DB写入
    """

    def run(self):
        """扫描并重试失败的同步"""
        pending_docs = self.pg.find("documents", {
            "status": "vector_pending",
            "retry_count": {"$lt": SagaCoordinator.MAX_RETRIES},
        })

        for doc in pending_docs:
            try:
                chunks = self.pg.find("chunks", {"doc_id": doc["id"]})
                embeddings = self.embedding_model.encode(
                    [c["content"] for c in chunks]
                )
                self.vector_db.upsert("documents", [
                    {"id": c["id"], "vector": emb.tolist(), "doc_id": doc["id"]}
                    for c, emb in zip(chunks, embeddings)
                ])

                # 成功，更新状态
                self.pg.update("chunks", {"doc_id": doc["id"]}, {
                    "sync_status": "synced"
                })
                self.pg.update("documents", {"id": doc["id"]}, {
                    "status": "active",
                    "last_error": None,
                })

            except Exception as e:
                self.pg.update("documents", {"id": doc["id"]}, {
                    "retry_count": doc.get("retry_count", 0) + 1,
                    "last_error": str(e)[:500],
                })

                if doc.get("retry_count", 0) >= SagaCoordinator.MAX_RETRIES - 1:
                    # 发送告警
                    alert(f"Vector sync failed after max retries for doc {doc['id']}")
```

### 11.4.4 冲突解决策略

```
冲突类型与解决策略矩阵:

┌─────────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ 冲突场景            │ LWW (Last-Write- │ 合并 (Merge)    │ 手动解决          │
│                     │ Wins)            │                  │ (Manual)         │
├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 同一文档并发更新     │ 版本号大者胜出   │ 三方合并          │ 标记冲突，        │
│                     │ 小者被拒绝       │ (base/ours/theirs)│ 人工干预          │
├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 文档删除 vs 更新    │ 时间戳晚者胜出   │ N/A               │ 人工确认          │
├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 向量版本不匹配      │ 以PG为准重新生成 │ N/A               │ N/A               │
├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 缓存与DB不一致      │ 以DB为准失效缓存 │ N/A               │ N/A               │
├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 分区间向量副本冲突  │ 向量相似度>0.99  │ 保留两个版本的    │ 标记后人工        │
│                     │ 则视为重复删除   │ 向量 + 优先级标记 │ 审核              │
└─────────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

### 11.4.5 灾备与跨区复制

```
跨区域灾备架构:

┌────────────────── Region A (Primary) ──────────────────┐
│                                                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐   │
│  │ PG       │   │ Milvus   │   │ Elasticsearch    │   │
│  │ (主库)   │   │ (主集群) │   │ (主集群)         │   │
│  └────┬─────┘   └────┬─────┘   └────────┬─────────┘   │
│       │              │                  │              │
│       │    ┌─────────┴─────────┐        │              │
│       │    │  Kafka MirrorMaker│        │              │
│       │    │  (跨区事件复制)    │        │              │
│       │    └─────────┬─────────┘        │              │
│       │              │                  │              │
│       │    ┌─────────┴─────────┐        │              │
│       │    │  PG 流复制        │        │              │
│       │    │  (WAL Shipping)   │        │              │
│       │    └─────────┬─────────┘        │              │
│       │              │                  │              │
└───────┼──────────────┼──────────────────┼──────────────┘
        │              │                  │
        ▼              ▼                  ▼
┌────────────────── Region B (Standby) ─────────────────┐
│                                                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐   │
│  │ PG       │   │ Milvus   │   │ Elasticsearch    │   │
│  │ (从库)   │   │ (从集群) │   │ (从集群)         │   │
│  └──────────┘   └──────────┘   └──────────────────┘   │
│                                                        │
└────────────────────────────────────────────────────────┘

备份策略:
┌───────────────────────────────────────────────────────┐
│ 存储层         │ 备份方式        │ RPO      │ RTO      │
├────────────────┼─────────────────┼──────────┼──────────┤
│ PostgreSQL     │ WAL连续归档     │ <1秒     │ <5分钟   │
│                │ + 每日全量快照   │          │          │
├────────────────┼─────────────────┼──────────┼──────────┤
│ 向量数据库     │ 定期快照导出    │ <1小时   │ <30分钟  │
│                │ + Kafka事件回放  │          │          │
├────────────────┼─────────────────┼──────────┼──────────┤
│ Elasticsearch  │ 快照到S3        │ <1小时   │ <30分钟  │
│                │ + 从PG重建       │          │          │
├────────────────┼─────────────────┼──────────┼──────────┤
│ Redis          │ RDB + AOF       │ <1分钟   │ <1分钟   │
│                │ 缓存可重建       │          │          │
└────────────────┴─────────────────┴──────────┴──────────┘
```

### 11.4.6 监控仪表盘

```python
"""
同步健康监控
"""
class SyncHealthMonitor:
    """
    同步健康度监控

    监控指标：
    - sync_lag_seconds: 从 PG 写入到向量DB可检索的延迟
    - orphan_vector_count: 孤儿向量数量
    - consistency_score: 一致性评分 (0-100)
    - pending_sync_count: 待同步记录数
    - error_rate: 同步错误率
    """

    def gather_metrics(self) -> Dict:
        """收集所有监控指标"""
        return {
            "sync_lag_seconds": self._measure_sync_lag(),
            "orphan_vector_count": self._count_orphans(),
            "pending_sync_count": self._count_pending(),
            "consistency_score": self._calculate_consistency_score(),
            "error_rate_5min": self._calculate_error_rate(minutes=5),
            "vector_db_latency_p99": self._measure_vector_latency(),
        }

    def _measure_sync_lag(self) -> float:
        """
        测量同步延迟

        方法：在PG中写入一条测试记录的时间戳，
             查询它何时在向量DB中变得可检索
        """
        test_id = f"__sync_lag_test_{uuid.uuid4().hex[:8]}"
        written_at = datetime.utcnow()

        # 写入测试记录
        self.pg.insert("sync_lag_tests", {
            "id": test_id,
            "written_at": written_at.isoformat(),
        })

        # 等待在消费者中出现
        timeout = 30  # 秒
        start = time.time()
        while time.time() - start < timeout:
            if self.vector_db.exists(test_id):
                lag = (datetime.utcnow() - written_at).total_seconds()
                # 清理
                self.pg.delete("sync_lag_tests", {"id": test_id})
                self.vector_db.delete(test_id)
                return lag
            time.sleep(0.1)

        return float("inf")  # 超时

    def _calculate_consistency_score(self) -> float:
        """
        计算一致性评分 (0-100)

        影响因素：
        - 计数一致性: 30分
        - 同步延迟: 30分 (lag<1s:30, lag<5s:20, lag>10s:0)
        - 孤儿向量: 20分 (0个:20, >100:0)
        - 待同步数: 20分 (0个:20, >50:0)
        """
        score = 0

        # 计数一致性检查
        checker = IntegrityChecker(self.pg, self.vector_db)
        count_check = checker.check_counts()
        if count_check["status"] == "pass":
            score += 30

        # 同步延迟
        lag = self._measure_sync_lag()
        if lag < 1:
            score += 30
        elif lag < 5:
            score += 20
        elif lag < 10:
            score += 10

        # 孤儿向量
        orphans = self._count_orphans()
        if orphans == 0:
            score += 20
        elif orphans < 50:
            score += 10

        # 待同步
        pending = self._count_pending()
        if pending == 0:
            score += 20
        elif pending < 10:
            score += 10

        return score
```

---

## 11.5 实现设计

### 11.5.1 数据库Schema设计（关系型+向量混合）

```sql
-- ============================================================
-- 知识库核心Schema (PostgreSQL 15+)
-- ============================================================

-- 1. 文档主表
CREATE TABLE documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID NOT NULL,                    -- 多租户
    knowledge_base_id UUID NOT NULL,                  -- 所属知识库
    title           VARCHAR(1024) NOT NULL,
    content         TEXT,
    content_hash    VARCHAR(64) NOT NULL,             -- SHA-256
    file_type       VARCHAR(32),                      -- pdf, docx, txt, md
    file_path       VARCHAR(2048),
    file_size       BIGINT DEFAULT 0,
    status          VARCHAR(32) DEFAULT 'active',     -- active/deleted/archived/pending
    version         INTEGER DEFAULT 1,
    chunk_count     INTEGER DEFAULT 0,
    sync_status     VARCHAR(32) DEFAULT 'synced',     -- synced/vector_pending/index_pending
    metadata        JSONB DEFAULT '{}',
    tags            TEXT[] DEFAULT ARRAY[]::TEXT[],
    deleted_at      TIMESTAMP,
    recovery_deadline TIMESTAMP,
    created_by      VARCHAR(128),
    updated_by      VARCHAR(128),
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW(),

    -- 约束
    CONSTRAINT chk_status CHECK (
        status IN ('active', 'deleted', 'archived', 'pending', 'processing')
    ),
    CONSTRAINT chk_sync_status CHECK (
        sync_status IN ('synced', 'vector_pending', 'index_pending', 'cache_stale')
    )
);

-- 2. 块表
CREATE TABLE chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id          UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,                -- 在文档中的序号
    content         TEXT NOT NULL,
    content_hash    VARCHAR(64) NOT NULL,
    token_count     INTEGER DEFAULT 0,
    status          VARCHAR(32) DEFAULT 'active',
    sync_status     VARCHAR(32) DEFAULT 'synced',
    vector_id       VARCHAR(256),                    -- 向量DB中的ID
    version         INTEGER DEFAULT 1,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW(),

    UNIQUE (doc_id, chunk_index)
);

-- 3. 同步状态表（用于补偿和追踪）
CREATE TABLE sync_log (
    id              BIGSERIAL PRIMARY KEY,
    doc_id          UUID REFERENCES documents(id),
    chunk_id        UUID REFERENCES chunks(id),
    target          VARCHAR(32) NOT NULL,            -- vector_db/search_index/cache
    operation       VARCHAR(32) NOT NULL,            -- INSERT/UPDATE/DELETE
    status          VARCHAR(32) DEFAULT 'pending',   -- pending/processing/completed/failed
    error_message   TEXT,
    retry_count     INTEGER DEFAULT 0,
    max_retries     INTEGER DEFAULT 5,
    next_retry_at   TIMESTAMP,
    completed_at    TIMESTAMP,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- 4. 补偿任务表
CREATE TABLE compensation_tasks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_type       VARCHAR(64) NOT NULL,
    target_id       UUID NOT NULL,
    target_type     VARCHAR(32) NOT NULL,            -- document/chunk/vector
    status          VARCHAR(32) DEFAULT 'pending',
    error_info      JSONB DEFAULT '{}',
    retry_count     INTEGER DEFAULT 0,
    max_retries     INTEGER DEFAULT 5,
    created_at      TIMESTAMP DEFAULT NOW(),
    next_retry_at   TIMESTAMP DEFAULT NOW(),
    completed_at    TIMESTAMP
);

-- 5. 操作审计表
CREATE TABLE audit_log (
    id              BIGSERIAL PRIMARY KEY,
    doc_id          UUID,
    chunk_id        UUID,
    operation       VARCHAR(64) NOT NULL,            -- CREATE/UPDATE/DELETE/RECOVER/ROLLBACK
    operator        VARCHAR(128),
    old_values      JSONB,
    new_values      JSONB,
    ip_address      INET,
    user_agent      VARCHAR(512),
    created_at      TIMESTAMP DEFAULT NOW()
);

-- 索引优化
CREATE INDEX idx_documents_status ON documents(status) WHERE status = 'active';
CREATE INDEX idx_documents_tenant ON documents(tenant_id, knowledge_base_id);
CREATE INDEX idx_documents_content_hash ON documents(content_hash);
CREATE INDEX idx_documents_sync_status ON documents(sync_status)
    WHERE sync_status != 'synced';
CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX idx_chunks_sync_status ON chunks(sync_status)
    WHERE sync_status != 'synced';
CREATE INDEX idx_sync_log_pending ON sync_log(status, next_retry_at)
    WHERE status IN ('pending', 'failed');
CREATE INDEX idx_audit_log_doc_id ON audit_log(doc_id, created_at DESC);

-- 全文搜索索引
CREATE INDEX idx_documents_fts ON documents
    USING GIN (to_tsvector('simple', coalesce(title, '') || ' ' || coalesce(content, '')));
```

### 11.5.2 API设计

```python
"""
知识库 CRUD API 设计 (FastAPI)
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


app = FastAPI(title="Enterprise Knowledge Base API", version="2.0.0")


# ========== 请求/响应模型 ==========

class DocCreateRequest(BaseModel):
    """创建文档请求"""
    title: str = Field(..., min_length=1, max_length=1024)
    content: str = Field(..., min_length=1)
    file_type: Optional[str] = "txt"
    knowledge_base_id: str
    tags: List[str] = []
    metadata: dict = {}

class DocUpdateRequest(BaseModel):
    """更新文档请求"""
    title: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[dict] = None

class DeleteMode(str, Enum):
    soft = "soft"
    hard = "hard"

class SyncStatus(str, Enum):
    synced = "synced"
    pending = "pending"
    failed = "failed"


# ========== API 端点 ==========

@app.post("/api/v2/knowledge-bases/{kb_id}/documents",
          response_model=DocResponse,
          status_code=201)
async def create_document(
    kb_id: str,
    request: DocCreateRequest,
    background_tasks: BackgroundTasks,
):
    """
    创建文档

    同步保证：
    - 返回 201 时，PG中的文档和块已持久化
    - 向量和索引通过后台异步同步
    - 可通过 GET /documents/{id}/sync-status 查询同步进度
    """
    doc = await document_service.create(request)
    # 异步触发向量化和索引
    background_tasks.add_task(sync_service.sync_new_document, doc.id)
    return doc

@app.get("/api/v2/documents/{doc_id}")
async def get_document(doc_id: str):
    """获取文档详情（包含同步状态）"""
    return await document_service.get_with_sync_status(doc_id)

@app.put("/api/v2/documents/{doc_id}")
async def update_document(
    doc_id: str,
    request: DocUpdateRequest,
    background_tasks: BackgroundTasks,
):
    """
    更新文档

    增量更新流程：
    1. 检测实际变更内容
    2. 仅重新处理变更部分
    3. 后台同步向量和索引
    """
    result = await document_service.update_incremental(doc_id, request)
    if result["affected_chunks"] > 0:
        background_tasks.add_task(
            sync_service.sync_updated_document,
            doc_id,
            result["affected_chunk_ids"],
        )
    return result

@app.delete("/api/v2/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    mode: DeleteMode = DeleteMode.soft,
    background_tasks: BackgroundTasks,
):
    """
    删除文档

    默认软删除（30天恢复窗口）
    hard模式：级联物理删除
    """
    result = await document_service.delete(doc_id, mode)
    background_tasks.add_task(
        sync_service.sync_deleted_document,
        doc_id,
        result["affected_chunks"],
    )
    return result

@app.post("/api/v2/documents/{doc_id}/recover")
async def recover_document(doc_id: str):
    """恢复软删除的文档"""
    return await document_service.recover(doc_id)

@app.get("/api/v2/documents/{doc_id}/sync-status")
async def get_sync_status(doc_id: str) -> SyncStatusResponse:
    """
    查询文档在各层的同步状态

    返回每个存储层的同步状态：
    - pg: synced
    - vector_db: synced / pending / failed
    - search_index: synced / pending / failed
    - cache: valid / stale
    """
    return await sync_service.get_document_sync_status(doc_id)

@app.post("/api/v2/documents/{doc_id}/rollback")
async def rollback_document(doc_id: str, target_version: int):
    """回滚文档到指定版本"""
    return await document_service.rollback(doc_id, target_version)

@app.get("/api/v2/knowledge-bases/{kb_id}/health")
async def kb_health_check(kb_id: str):
    """知识库健康检查"""
    return await health_service.check_kb_health(kb_id)

@app.post("/api/v2/admin/integrity-check")
async def run_integrity_check(background_tasks: BackgroundTasks):
    """管理员触发完整性检查"""
    task_id = str(uuid.uuid4())
    background_tasks.add_task(integrity_service.run_full_check, task_id)
    return {"task_id": task_id, "status": "started"}
```

### 11.5.3 后台同步工作器架构

```
┌─────────────────────────────────────────────────────────────┐
│               后台同步工作器架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Worker Manager (Supervisor)              │   │
│  │  - 工作器生命周期管理                                │   │
│  │  - 健康检查和自动重启                                │   │
│  │  - 负载均衡                                          │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│     ┌───────────────────┼───────────────────┐              │
│     ▼                   ▼                   ▼              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ Worker 1 │    │  Worker 2    │    │  Worker 3    │     │
│  │          │    │              │    │              │     │
│  │ Vector   │    │  Search      │    │  Cache       │     │
│  │ Syncer   │    │  Index       │    │  Invalidation│     │
│  │          │    │  Syncer      │    │  Worker      │     │
│  │ Consumes:│    │  Consumes:   │    │  Consumes:   │     │
│  │ Kafka    │    │  Kafka       │    │  Kafka       │     │
│  └──────────┘    └──────────────┘    └──────────────┘     │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Scheduled Workers (Cron)                 │  │
│  │                                                      │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │  │
│  │  │ Orphan      │  │ Compensation │  │ Integrity  │  │  │
│  │  │ Scanner     │  │ Worker       │  │ Checker    │  │  │
│  │  │ (每6小时)   │  │ (每30秒)     │  │ (每日)     │  │  │
│  │  └─────────────┘  └──────────────┘  └────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 11.5.4 并发更新锁定策略

```python
"""
并发更新锁定策略
"""
from contextlib import contextmanager
import redis


class ConcurrencyControl:
    """
    并发控制

    策略层次：
    1. 乐观锁 (Optimistic Locking) - 默认
    2. 悲观锁 (Pessimistic Locking) - 高冲突场景
    3. 分布式锁 (Distributed Lock) - 跨服务操作
    """

    def __init__(self, pg_client, redis_client: redis.Redis):
        self.pg = pg_client
        self.redis = redis_client

    # ---- 策略1: 乐观锁 ----

    def update_with_optimistic_lock(
        self, doc_id: str, updates: Dict, expected_version: int
    ) -> Dict:
        """
        乐观锁更新

        如果 expected_version != 当前版本，说明有并发更新，拒绝操作
        """
        result = self.pg.execute("""
            UPDATE documents
            SET title = COALESCE(%(title)s, title),
                content = COALESCE(%(content)s, content),
                version = version + 1,
                updated_at = NOW()
            WHERE id = %(doc_id)s
              AND version = %(expected_version)s
            RETURNING version
        """, {
            "doc_id": doc_id,
            "title": updates.get("title"),
            "content": updates.get("content"),
            "expected_version": expected_version,
        })

        if result.rowcount == 0:
            # 版本不匹配，可能被并发更新
            current = self.pg.find_one("documents", {"id": doc_id})
            raise ConcurrencyConflictError(
                f"Version conflict: expected {expected_version}, "
                f"current {current['version'] if current else 'deleted'}"
            )

        return {"doc_id": doc_id, "new_version": expected_version + 1}

    # ---- 策略2: 悲观锁 ----

    @contextmanager
    def pessimistic_lock(self, doc_id: str):
        """
        悲观锁（行级锁）

        在事务中使用 SELECT ... FOR UPDATE
        """
        with self.pg.transaction() as tx:
            self.pg.execute(
                "SELECT id FROM documents WHERE id = %s FOR UPDATE",
                (doc_id,)
            )
            try:
                yield tx
                tx.commit()
            except Exception:
                tx.rollback()
                raise

    # ---- 策略3: 分布式锁 ----

    @contextmanager
    def distributed_lock(
        self, doc_id: str, ttl_seconds: int = 30
    ):
        """
        分布式锁 (基于Redis)

        用于跨服务实例的并发控制
        """
        lock_key = f"lock:doc:{doc_id}"
        lock_value = str(uuid.uuid4())
        acquired = self.redis.set(
            lock_key, lock_value, nx=True, ex=ttl_seconds
        )

        if not acquired:
            raise LockAcquisitionError(
                f"Failed to acquire lock for document {doc_id}"
            )

        try:
            yield
        finally:
            # 使用 Lua 脚本安全释放锁（仅释放自己持有的锁）
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            self.redis.eval(lua_script, 1, lock_key, lock_value)

    # ---- 选择合适的策略 ----

    def auto_select_strategy(
        self, doc_id: str
    ) -> str:
        """
        自动选择锁策略

        根据文档的访问模式和冲突历史自动选择
        """
        conflict_rate = self._get_conflict_rate(doc_id)

        if conflict_rate > 0.1:  # 冲突率 > 10%
            return "pessimistic"
        elif conflict_rate > 0.01:  # 冲突率 > 1%
            return "optimistic"
        else:
            return "none"  # 低冲突，无需锁

    def _get_conflict_rate(self, doc_id: str) -> float:
        """计算文档的历史冲突率"""
        stats = self.pg.find_one("doc_lock_stats", {"doc_id": doc_id})
        if not stats:
            return 0.0
        total = stats.get("total_updates", 0)
        conflicts = stats.get("conflicts", 0)
        return conflicts / max(total, 1)
```

---

## 11.6 面试题精选

### Q1: 在RAG系统中，用户删除了一篇文档，但检索时仍然能搜到相关内容，可能的原因有哪些？如何排查和修复？

**参考答案**：

**可能原因**（按概率排序）：
1. **向量数据库未同步删除**：最常见原因。文档在PG中删除，但向量残留。排查：直接查询向量DB `count where doc_id = X`。
2. **缓存未失效**：Redis中缓存了旧的检索结果。排查：检查是否有 `tombstone` 标记键。
3. **搜索引擎索引未更新**：Elasticsearch中仍有该文档的索引条目。
4. **软删除未过滤**：查询时未加 `WHERE status != 'deleted'` 条件。
5. **异步消费者延迟/故障**：Kafka消费者处理落后或崩溃。
6. **删除操作部分失败**：块删除了但向量删除失败，无补偿机制。

**排查步骤**：
```
1. 检查PG中文档状态: SELECT status FROM documents WHERE id = X
2. 检查向量DB: 直接调用向量DB API查询
3. 检查Redis tombstone: GET tombstone:doc:X
4. 检查Kafka消费者组 lag
5. 检查 sync_log 表
6. 检查补偿任务表
```

**修复方案**：
- 短期：手动调用清理API
- 长期：完善级联删除+补偿任务+监控告警

### Q2: 如何设计一个支持亿级文档的知识库同步系统？

**参考答案**：

核心设计要点：
1. **分片策略**：按 tenant_id + kb_id 哈希分片，每个分片独立同步
2. **Kafka分区**：按 doc_id 哈希分区，保证同一文档事件有序
3. **批量处理**：消费者攒批处理，批量写入向量DB
4. **背压机制**：当消费者处理不过来时，降低生产速率
5. **分级存储**：热数据全量同步，冷数据延迟同步
6. **并行度控制**：消费者组实例数 = Kafka分区数

### Q3: 向量数据库和关系型数据库的数据不一致时，以谁为准？

**参考答案**：

**以关系型数据库（PostgreSQL）为真相之源（Source of Truth）**。

原因：
1. PG支持ACID事务，数据可靠性更高
2. 向量可以从原始文本重新生成
3. PG有完整的WAL日志和备份机制
4. 向量DB主要用于相似性搜索，其精度需求相对宽松

不一致时的处理策略：
```
if PG有记录 and 向量DB无记录:
    → 重新生成向量（补偿同步）
if PG无记录 and 向量DB有记录:
    → 清理孤儿向量
if PG和向量DB都有，但内容不一致:
    → 以PG为准，重新向量化
```

### Q4: 已有1000万文档的知识库，Embedding模型升级了，如何最小化业务影响完成迁移？

**参考答案**：

使用 **滚动重建（Rolling Rebuild）+ 版本化查询** 策略：

```
阶段1: 准备期（零影响）
- 部署新模型
- 创建新的向量集合（v2），启用双写

阶段2: 迁移期（部分影响）
- 后台任务逐步将旧向量重新编码为新模型向量
- 查询时：优先从v2检索，v2无结果时fallback到v1
- 渐进式切换：按知识库粒度逐步切换

阶段3: 清理期（零影响）
- 所有查询走v2
- 删除v1集合
```

### Q5: 设计一个文档更新的"最后写入胜利"(LWW)冲突解决策略，有哪些潜在的陷阱？

**参考答案**：

LWW的陷阱：
1. **时钟不同步**：分布式节点的时钟偏差导致"后写"实际是先发生的变更
2. **静默丢失**：并发更新中，先提交的变更被后提交的完全覆盖，无任何通知
3. **向量版本错乱**：LWW只解决了文本冲突，但向量可能基于中间版本生成
4. **缓存不一致**：LWW提交后，缓存中可能还残留旧版本数据

改进建议：
- 使用 **向量时钟（Vector Clock）** 替代物理时间戳
- 版本号强制单调递增（利用PG的序列）
- 并发冲突时保存冲突副本，通知用户
- 写入成功后主动失效所有相关缓存

---

## 11.7 企业最佳实践总结

### 实践清单

```
┌─────────────────────────────────────────────────────────────────┐
│                    企业级CRUD同步最佳实践清单                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [设计原则]                                                      │
│  ☑ 1. 以关系型数据库为唯一真相之源 (Single Source of Truth)       │
│  ☑ 2. 异步最终一致性优先于强一致性（除非合规要求）                  │
│  ☑ 3. 所有操作幂等：同一操作执行多次结果相同                       │
│  ☑ 4. 写操作记录完整审计日志                                      │
│                                                                 │
│  [实现规范]                                                      │
│  ☑ 5. 文档删除必须级联：文档→块→向量→索引→缓存                     │
│  ☑ 6. 默认软删除，30天恢复窗口                                    │
│  ☑ 7. 更新操作优先使用增量处理（仅处理变更块）                      │
│  ☑ 8. 所有写操作通过Kafka事件总线分发                             │
│  ☑ 9. 消费者实现至少一次处理语义 + 幂等写入                        │
│  ☑ 10. 补偿任务表 + 定时扫描，保证最终一致性                       │
│                                                                 │
│  [运维保障]                                                      │
│  ☑ 11. 定期孤儿向量扫描（每6小时）                                 │
│  ☑ 12. 每日全量完整性检查                                         │
│  ☑ 13. 同步延迟监控（P50/P95/P99）+ 告警                         │
│  ☑ 14. 一致性评分仪表盘                                           │
│  ☑ 15. 跨区域灾备 + 演练                                          │
│                                                                 │
│  [性能优化]                                                      │
│  ☑ 16. 向量化批量处理（batch_size=32-64）                        │
│  ☑ 17. 缓存预热：热点文档预加载向量到内存                          │
│  ☑ 18. 索引滚动重建，避免停机                                     │
│  ☑ 19. 落盘前攒批（linger.ms=5）减少网络开销                     │
│  ☑ 20. 向量量化压缩（PQ/Scalar Quantization）减少存储            │
│                                                                 │
│  [安全合规]                                                      │
│  ☑ 21. 所有操作记录审计日志（谁、何时、做了什么）                   │
│  ☑ 22. 敏感文档加密存储                                          │
│  ☑ 23. API访问限流 + 鉴权                                        │
│  ☑ 24. PII检测 + 脱敏处理                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 核心原则回顾

1. **真相之源原则**：PostgreSQL是数据的唯一权威来源。向量数据库、搜索引擎、缓存都是真相之源的投影，可以随时从PG重建。

2. **最终一致性原则**：除非业务强需求（如金融合规），否则追求最终一致性。通过异步事件+补偿任务保证各层最终达到一致。

3. **防腐蚀原则**：系统需要主动防御数据退化，而非被动修复。孤儿扫描、完整性校验、变更事件验证等机制应作为基础设施运行。

4. **可观测性原则**：每一层的一致性状态必须可度量、可监控、可告警。不能"相信"系统是一致的，而要"验证"系统是一致的。

---

*本章完。第11章覆盖了知识库CRUD全生命周期操作、多存储层一致性机制、防腐蚀策略、完整同步架构以及企业级实现设计。核心回答了"在原始文档、向量数据库、缓存、索引、关系型数据库之间如何维护一致性"这一企业级RAG系统的关键命题。*
