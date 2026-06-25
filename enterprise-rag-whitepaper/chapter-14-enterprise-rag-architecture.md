# 第十四章 企业级RAG项目架构设计

## 章节概述

企业级RAG（Retrieval-Augmented Generation）系统的架构设计是一项复杂的系统工程，涉及计算资源规划、分布式系统部署、高可用保障、安全合规和可观测性等多个维度。本章从百万级文档到千万级文档的规模跨度，系统性地阐述RAG系统的架构设计方法论，为技术决策者和架构师提供可落地的参考方案。

**核心目标**：
- 建立从百万级到千万级文档的RAG系统容量规划模型
- 提供完整的云原生部署架构方案（Kubernetes + 服务网格）
- 构建全方位的监控、告警和可观测性体系
- 实现企业级安全架构（多租户隔离、数据合规、审计追溯）

---

## 第一节：百万级文档RAG系统设计

### 1.1 概念定义与背景

**百万级文档RAG系统**是指能够承载100万至500万份文档的检索增强生成平台。以每份文档平均10页（约5000字符）计算，百万级文档意味着：
- 原始文档总量：约5TB（含PDF、Word、Markdown等格式）
- 文档切片（Chunk）数量：约5000万至2亿条（每文档50-200个切片）
- 向量数据规模：约200GB至800GB（以768维float32计算）
- 日均查询量（QPS）：100-1000 QPS

此类系统适用于中大型企业的知识管理、智能客服、合规审查、研发知识库等场景。

### 1.2 容量规划

#### 1.2.1 存储估算模型

```
存储估算公式：
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  原始存储 = 文档数量 × 平均文档大小 × 存储副本系数              │
│           = 1,000,000 × 5MB × 1.2（元数据冗余）                 │
│           ≈ 6 TB                                               │
│                                                                 │
│  块存储   = 文档数量 × 块/文档 × 块大小 × (1 + 元数据比例)      │
│           = 1,000,000 × 100 × 500 bytes × 1.3                  │
│           ≈ 65 GB                                              │
│                                                                 │
│  向量存储 = 块数量 × 向量维度 × 4 bytes × 索引倍数              │
│           = 100,000,000 × 768 × 4 × 1.5（HNSW索引开销）         │
│           ≈ 460 GB                                             │
│                                                                 │
│  总存储   = 原始存储 + 块存储 + 向量存储 + 日志/缓存             │
│           ≈ 6 TB + 65 GB + 460 GB + 500 GB                     │
│           ≈ 7 TB（含20%增长预留 ≈ 8.5 TB）                      │
└─────────────────────────────────────────────────────────────────┘
```

#### 1.2.2 计算资源估算

| 组件 | 计算方式 | 百万级估算 | 备注 |
|------|----------|-----------|------|
| Embedding服务 | QPS × 单文档Chunk数 × GPU耗时 | 4×A100 (80GB) | 批处理优化后可减少 |
| 向量检索 | 向量数 × log(向量数) × 单次计算 | 32 vCPU + 128GB RAM | 依赖索引类型 |
| Rerank服务 | QPS × 候选数 × 模型耗时 | 2×A10 (24GB) | Cross-encoder模型 |
| LLM推理 | QPS × 平均Token数 / GPU吞吐 | (4-8)×A100 (80GB) | vLLM连续批处理 |
| 缓存服务 | 缓存命中率 × QPS × 响应大小 | 16 vCPU + 64GB RAM | Redis集群 |
| API网关 | QPS × (认证+限流+路由) | 8 vCPU + 16GB RAM | Kong/APISIX |

### 1.3 完整架构图

下面是百万级文档RAG系统的完整架构拓扑（详细的ASCII架构图）：

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              百万级文档RAG系统 - 完整架构拓扑                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │   CDN / DNS  │
                              │  (CloudFront │
                              │   / 阿里CDN) │
                              └──────┬───────┘
                                     │
                            ┌────────▼────────┐
                            │   负载均衡器     │
                            │  (Nginx/HAProxy)│
                            │  · Round-Robin  │
                            │  · Health Check │
                            │  · TLS Termination│
                            └────────┬────────┘
                                     │
                      ┌──────────────▼──────────────┐
                      │       API 网关层             │
                      │   ┌──────────────────────┐  │
                      │   │  Kong / APISIX 集群  │  │
                      │   │  · 限流 (Rate Limit) │  │
                      │   │  · 认证 (JWT/OAuth)  │  │
                      │   │  · 路由分发          │  │
                      │   │  · 请求/响应转换     │  │
                      │   │  · 插件管理          │  │
                      │   └──────────────────────┘  │
                      └──────────────┬──────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
    ┌────▼─────┐              ┌──────▼──────┐            ┌──────▼──────┐
    │ 查询服务  │              │  管理服务    │            │  异步任务   │
    │ (Query)  │              │  (Admin)    │            │  (Async)    │
    │          │              │             │            │             │
    │ · 查询拆解│              │ · 文档上传   │            │ · 文档处理  │
    │ · 意图识别│              │ · 知识库管理 │            │ · 向量化   │
    │ · 检索编排│              │ · 用户管理   │            │ · 增量更新  │
    │ · 答案生成│              │ · 配置管理   │            │ · 质量检测  │
    └────┬─────┘              └──────┬──────┘            └──────┬──────┘
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     │
                         ┌───────────▼───────────┐
                         │     消息队列层         │
                         │  ┌─────────────────┐  │
                         │  │  Apache Kafka   │  │
                         │  │  · doc.uploaded │  │
                         │  │  · doc.parsed   │  │
                         │  │  · chunk.indexed│  │
                         │  │  · index.refresh│  │
                         │  └─────────────────┘  │
                         └───────────┬───────────┘
                                     │
    ┌────────────┬──────────┬────────┼────────┬──────────┬────────────┐
    │            │          │        │        │          │            │
┌───▼───┐  ┌────▼────┐ ┌──▼──┐ ┌──▼──┐ ┌──▼───┐ ┌───▼────┐ ┌────▼─────┐
│ 缓存层 │  │Embedding│ │向量  │ │Rerank│ │ LLM  │ │关系型   │ │对象存储  │
│       │  │  服务    │ │数据库 │ │服务  │ │ 服务 │ │数据库   │ │          │
│       │  │         │ │      │ │      │ │      │ │        │ │          │
│ Redis │  │ ┌─────┐ │ │Milvus│ │ ┌──┐ │ │┌───┐ │ │PostgreSQL│ │MinIO/S3 │
│Cluster│  │ │GPU 1│ │ │/Qdrant│ │ │BGE││ ││vLLM│ │ │        │ │          │
│       │  │ │GPU 2│ │ │      │ │ │Re-││ ││SGLang│ │ │· 用户表 │ │· 原始文档│
│·结果  │  │ │GPU 3│ │ │· 分片 │ │ │ran││ ││    │ │ │· 元数据 │ │· 版本控制│
│ 缓存  │  │ │GPU 4│ │ │· 副本 │ │ │ker││ ││· 连续│ │ │· 配置  │ │· 备份   │
│·嵌入  │  │ └─────┘ │ │· 索引 │ │ └──┘│ ││ 批处 │ │ │· 审计  │ │          │
│ 缓存  │  │         │ │      │ │      │ ││ 理  │ │ │        │ │          │
│·查询  │  │批处理队列│ │· 监控 │ │ 批处理│ │└───┘ │ │        │ │          │
│ 缓存  │  │         │ │      │ │      │ │      │ │        │ │          │
└───┬───┘  └────┬────┘ └──┬───┘ └──┬───┘ └──┬───┘ └───┬────┘ └────┬─────┘
    │            │          │        │        │          │            │
    └────────────┴──────────┴────────┴────────┴──────────┴────────────┘
                                     │
                         ┌───────────▼───────────┐
                         │      监控与可观测层     │
                         │  ┌─────────────────┐  │
                         │  │ Prometheus      │  │
                         │  │ + Grafana       │  │
                         │  │ + Jaeger 追踪   │  │
                         │  │ + ELK 日志      │  │
                         │  │ + AlertManager  │  │
                         │  └─────────────────┘  │
                         └───────────────────────┘
```

### 1.4 组件详细设计

#### 1.4.1 负载均衡器（Nginx/HAProxy）

**概念**：负载均衡器是整个系统的流量入口，负责将客户端请求均匀分配到后端的多个API网关实例，同时提供TLS终止和健康检查能力。

**工作原理**：
```
客户端请求 → DNS解析 → 负载均衡器VIP
                          │
              ┌───────────┼───────────┐
              │           │           │
         Nginx-1      Nginx-2     Nginx-3  (Active-Active)
              │           │           │
              └───────────┼───────────┘
                          │
                    API网关集群
```

**算法选择**：

| 算法 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| Round-Robin | 后端实例性能均匀 | 实现简单，开销低 | 无法应对性能差异 |
| Least-Connections | 长连接场景（LLM流式） | 动态适应负载 | 需要维护连接计数 |
| IP Hash | 会话保持需求 | 同一客户端路由一致 | 负载可能不均 |
| 加权轮询 | 后端实例配置不同 | 灵活分配 | 需要人工设置权重 |

**Nginx配置示例**：
```nginx
upstream api_gateway {
    least_conn;
    server api-gateway-1.internal:8080 weight=5 max_fails=3 fail_timeout=30s;
    server api-gateway-2.internal:8080 weight=5 max_fails=3 fail_timeout=30s;
    server api-gateway-3.internal:8080 weight=3 max_fails=3 fail_timeout=30s;
    
    keepalive 128;
}

upstream llm_service {
    least_conn;
    server llm-service-1.internal:8000 weight=5;
    server llm-service-2.internal:8000 weight=5;
    server llm-service-3.internal:8000 weight=5;
    
    # LLM请求通常为长连接
    keepalive 64;
    keepalive_timeout 300s;
}

server {
    listen 443 ssl http2;
    server_name rag-api.example.com;
    
    ssl_certificate     /etc/nginx/certs/fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    
    # 请求体大小限制（文档上传）
    client_max_body_size 100M;
    
    location /api/v1/query {
        proxy_pass http://api_gateway;
        proxy_read_timeout 120s;  # RAG查询可能较慢
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    location /api/v1/upload {
        proxy_pass http://api_gateway;
        proxy_read_timeout 300s;  # 文档上传超时更长
        client_max_body_size 100M;
    }
    
    location /api/v1/stream {
        proxy_pass http://llm_service;
        proxy_buffering off;      # SSE流式响应
        proxy_cache off;
        proxy_read_timeout 600s;
    }
}
```

#### 1.4.2 API网关（Kong/APISIX）

**概念**：API网关是微服务架构的统一入口，负责请求路由、认证授权、限流、日志采集和协议转换。在RAG系统中，网关承担着流量治理的核心角色。

**核心功能与插件配置**：

```
┌──────────────────────────────────────────────────┐
│                  API 网关 插件链                   │
├──────────────────────────────────────────────────┤
│                                                   │
│  请求进入 → [认证] → [限流] → [路由] → [转换] → 后端│
│                                                   │
│  认证插件:                                        │
│  · JWT Token 验证                                 │
│  · API Key 验证                                   │
│  · OAuth2.0 集成                                  │
│                                                   │
│  限流插件:                                        │
│  · 用户级: 100 req/min/user                       │
│  · IP级:   500 req/min/ip                         │
│  · 接口级: 1000 req/min/endpoint                   │
│                                                   │
│  路由配置 (APISIX):                               │
│  /api/v1/query     → query-service:8080           │
│  /api/v1/upload    → admin-service:8080           │
│  /api/v1/chat      → llm-service:8000             │
│  /api/v1/search    → retrieval-service:8081       │
│                                                   │
└──────────────────────────────────────────────────┘
```

**APISIX路由配置示例**：
```yaml
# APISIX 路由配置
routes:
  - id: rag-query
    uri: /api/v1/query
    upstream:
      type: roundrobin
      nodes:
        query-service-1:8080: 1
        query-service-2:8080: 1
    plugins:
      jwt-auth:
        key: user_key
      limit-req:
        rate: 30
        burst: 60
        key: http_x_user_id
      prometheus:
        prefer_name: true
      proxy-rewrite:
        headers:
          X-Service: "rag-query"
          
  - id: rag-stream
    uri: /api/v1/stream
    upstream:
      type: least_conn
      nodes:
        llm-service-1:8000: 1
        llm-service-2:8000: 1
    plugins:
      jwt-auth: {}
      limit-conn:
        conn: 50          # 流式接口连接数限制
        burst: 10
        default_conn_delay: 0.5

  - id: rag-upload
    uri: /api/v1/upload
    upstream:
      type: roundrobin
      nodes:
        admin-service-1:8080: 1
        admin-service-2:8080: 1
    plugins:
      jwt-auth: {}
      limit-req:
        rate: 10
        burst: 20
      file-logger:
        path: /var/log/apisix/upload.log
```

#### 1.4.3 缓存层（Redis集群）

**概念**：多级缓存是RAG系统性能优化的核心策略。通过在不同层级建立缓存，可以显著减少重复计算、降低LLM调用成本和缩短端到端延迟。

**多级缓存架构**：

```
┌──────────────────────────────────────────────────────┐
│                   多级缓存架构                         │
├──────────────────────────────────────────────────────┤
│                                                      │
│  L1: 结果缓存 (用户问题Hash → 完整答案)               │
│  ┌────────────────────────────────────────────┐      │
│  │ TTL: 1小时  │ 命中率: 10-15% │ 存储: 5GB   │      │
│  │ Key: md5(tenant_id + query + top_k + model) │      │
│  │ 适用: 热点问题（FAQ类）重复查询               │      │
│  └────────────────────────────────────────────┘      │
│                                                      │
│  L2: 嵌入向量缓存 (文本Hash → 向量)                   │
│  ┌────────────────────────────────────────────┐      │
│  │ TTL: 24小时 │ 命中率: 30-50% │ 存储: 20GB  │      │
│  │ Key: md5(tenant_id + chunk_text)            │      │
│  │ 适用: 文档未变时避免重复Embedding计算         │      │
│  └────────────────────────────────────────────┘      │
│                                                      │
│  L3: 语义缓存 (语义相似查询 → 相同答案)               │
│  ┌────────────────────────────────────────────┐      │
│  │ TTL: 2小时  │ 命中率: 15-20% │ 存储: 3GB   │      │
│  │ 相似度阈值: ≥0.95                            │      │
│  │ Key: 向量聚类中心点                          │      │
│  └────────────────────────────────────────────┘      │
│                                                      │
│  L4: LLM Token缓存 (Prompt Hash → 生成的Token)       │
│  ┌────────────────────────────────────────────┐      │
│  │ TTL: 30分钟 │ 命中率: 5-8%  │ 存储: 2GB    │      │
│  │ Key: md5(system_prompt + context + question)│      │
│  └────────────────────────────────────────────┘      │
│                                                      │
└──────────────────────────────────────────────────────┘
```

**Redis集群配置**：
```yaml
# Redis Cluster 配置
redis:
  mode: cluster
  nodes:
    - redis-node1:6379
    - redis-node2:6379
    - redis-node3:6379
    - redis-node4:6379
    - redis-node5:6379
    - redis-node6:6379
  replicas: 1  # 每个主节点1个副本
  config:
    maxmemory: "16gb"
    maxmemory-policy: "allkeys-lru"  # 内存满时淘汰最少使用
    save: "900 1 300 10 60 10000"   # RDB持久化策略
    appendonly: yes
    appendfsync: everysec
```

**缓存实现伪代码**：
```python
class MultiLevelCache:
    """RAG系统多级缓存管理器"""
    
    def __init__(self, redis_cluster):
        self.redis = redis_cluster
        self.similarity_threshold = 0.95
        
    async def get_query_result(self, tenant_id: str, query: str, 
                                top_k: int, model: str) -> Optional[str]:
        """L1: 获取缓存的结果"""
        cache_key = hashlib.md5(
            f"{tenant_id}:{query}:{top_k}:{model}".encode()
        ).hexdigest()
        return await self.redis.get(f"cache:result:{cache_key}")
    
    async def set_query_result(self, tenant_id: str, query: str,
                                top_k: int, model: str, result: str):
        """L1: 缓存查询结果"""
        cache_key = hashlib.md5(
            f"{tenant_id}:{query}:{top_k}:{model}".encode()
        ).hexdigest()
        await self.redis.setex(
            f"cache:result:{cache_key}",
            3600,  # TTL 1小时
            result
        )
    
    async def get_embedding(self, text: str) -> Optional[list]:
        """L2: 获取缓存的嵌入向量"""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        cached = await self.redis.get(f"cache:embed:{cache_key}")
        if cached:
            return json.loads(cached)
        return None
    
    async def find_similar_query(self, query_embedding: list) -> Optional[str]:
        """L3: 通过语义相似度查找缓存"""
        # 使用Redis向量搜索或LSH近似查找
        # 若相似度 >= 0.95, 返回缓存结果
        similar = await self.redis.ft_search(
            "idx:query_cache",
            f"@embedding:[VECTOR_RANGE 0.05 $vec]", 
            {"vec": query_embedding}
        )
        if similar and similar[0].score >= self.similarity_threshold:
            return similar[0].result
        return None
```

#### 1.4.4 Embedding服务

**概念**：Embedding服务负责将文本块转换为稠密向量表示，是RAG系统的计算密集型组件之一。GPU批处理是提升Embedding吞吐量的核心策略。

**架构设计**：
```
                    ┌─────────────────────┐
                    │   Embedding 请求队列  │
                    │  (Redis / RabbitMQ)  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    批处理器          │
                    │  (Batch Aggregator) │
                    │                     │
                    │  · 最大批次: 256    │
                    │  · 最大等待: 50ms   │
                    │  · 动态批大小调整    │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
        ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
        │  GPU 0    │   │  GPU 1    │   │  GPU 2    │
        │  A100-80G │   │  A100-80G │   │  A100-80G │
        │           │   │           │   │           │
        │ Model:    │   │ Model:    │   │ Model:    │
        │ BGE-M3    │   │ BGE-M3    │   │ BGE-M3    │
        │ Batch: 64 │   │ Batch: 64 │   │ Batch: 64 │
        └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
              │                │                │
              └────────────────┼────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │    结果聚合与缓存     │
                    │                     │
                    │  · 写入L2向量缓存    │
                    │  · 写入向量数据库     │
                    └─────────────────────┘
```

**模型选型对比**：

| 模型 | 维度 | 最大Token | 单GPU吞吐(条/s) | 中文效果 | MTEB得分 |
|------|------|-----------|----------------|---------|---------|
| BGE-M3 | 1024 | 8192 | ~800 | 优秀 | 68.3 |
| BGE-Large-ZH | 1024 | 512 | ~600 | 优秀 | 64.2 |
| text2vec-large-chinese | 1024 | 512 | ~500 | 良好 | 62.1 |
| stella-mrl-large-zh | 1024 | 512 | ~550 | 优秀 | 65.8 |
| multilingual-e5-large | 1024 | 512 | ~700 | 良好 | 66.5 |
| GTE-Qwen2-7B-instruct | 3584 | 32768 | ~300 | 极优 | 70.2 |

**批处理优化实现**：
```python
class EmbeddingService:
    """GPU批处理Embedding服务"""
    
    def __init__(self, model_name="BAAI/bge-m3", 
                 max_batch_size=256, 
                 max_wait_ms=50):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self.model.cuda().eval()
        
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending_queue = asyncio.Queue()
        self.batch_event = asyncio.Event()
        
    async def embed_async(self, texts: List[str]) -> np.ndarray:
        """异步批量嵌入"""
        # 先检查L2缓存
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cached = await cache.get_embedding(text)
            if cached:
                embeddings.append((i, cached))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        if uncached_texts:
            # 动态批处理
            batch_embeddings = await self._batch_process(uncached_texts)
            for idx, emb in zip(uncached_indices, batch_embeddings):
                embeddings.append((idx, emb))
        
        # 按原始顺序返回
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])
    
    async def _batch_process(self, texts: List[str]) -> np.ndarray:
        """动态批处理 - 累积到最大批大小或超时"""
        if len(texts) >= self.max_batch_size:
            return self._encode_batch(texts[:self.max_batch_size])
        
        # 分批处理大列表
        results = []
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            results.append(self._encode_batch(batch))
        
        return np.concatenate(results)
    
    @torch.no_grad()
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """GPU编码一批文本"""
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=8192,
            return_tensors="pt"
        ).to("cuda")
        
        outputs = self.model(**inputs)
        # 使用[CLS] token或均值池化
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy().astype(np.float32)
```

#### 1.4.5 向量数据库集群

**概念**：向量数据库是RAG系统的核心存储组件，负责存储和检索海量文档块对应的向量表示。生产环境需要支持分片（Sharding）和副本（Replication）以实现高可用和水平扩展。

**技术选型对比**：

| 特性 | Milvus | Qdrant | Weaviate | Elasticsearch | Chroma |
|------|--------|--------|----------|---------------|--------|
| 分布式架构 | 原生支持 | 原生支持 | 原生支持 | 原生支持 | 单机 |
| 数据分片 | 自动Hash | 用户定义 | 自动 | 基于索引 | 不支持 |
| 索引算法 | IVF/HNSW/DiskANN | HNSW | HNSW | HNSW | HNSW |
| 过滤查询 | 标量+向量混合 | 负载过滤 | GraphQL过滤 | DSL查询 | 基础过滤 |
| 一致性 | 可调 | 可调 | 最终一致 | 可调 | N/A |
| GPU加速 | 支持 | 不支持 | 不支持 | 不支持 | 不支持 |
| 监控集成 | Prometheus | Prometheus | Prometheus | Elastic Stack | 有限 |
| 百万级推荐 | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★☆☆☆ |

**Milvus集群架构**：
```
┌─────────────────────────────────────────────┐
│              Milvus 分布式集群                │
├─────────────────────────────────────────────┤
│                                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │  Proxy  │  │  Proxy  │  │  Proxy  │      │
│  │  Node 1 │  │  Node 2 │  │  Node 3 │      │
│  └────┬────┘  └────┬────┘  └────┬────┘      │
│       │            │            │            │
│       └────────────┼────────────┘            │
│                    │                         │
│       ┌────────────┼────────────┐            │
│       │            │            │            │
│  ┌────▼────┐  ┌────▼────┐  ┌───▼─────┐     │
│  │  Query  │  │  Query  │  │  Query  │      │
│  │  Node 1 │  │  Node 2 │  │  Node 3 │      │
│  └────┬────┘  └────┬────┘  └────┬────┘      │
│       │            │            │            │
│       └────────────┼────────────┘            │
│                    │                         │
│  ┌─────────────────┼─────────────────┐       │
│  │                 │                 │       │
│  ▼                 ▼                 ▼       │
│ ┌──────┐      ┌──────┐        ┌──────┐      │
│ │ Data │      │ Data │        │ Data │      │
│ │Node 1│      │Node 2│        │Node 3│      │
│ │      │      │      │        │      │      │
│ │Shard │      │Shard │        │Shard │      │
│ │ 0(主)│      │ 1(主)│        │ 2(主)│      │
│ │ 1(副)│      │ 2(副)│        │ 0(副)│      │
│ └──────┘      └──────┘        └──────┘      │
│                                              │
│  ┌──────────────────────────────┐            │
│  │     元数据存储 (etcd)          │            │
│  │     · 集群拓扑                 │            │
│  │     · 节点状态                 │            │
│  │     · 配置管理                 │            │
│  └──────────────────────────────┘            │
│                                              │
│  ┌──────────────────────────────┐            │
│  │     对象存储 (MinIO/S3)        │            │
│  │     · 向量索引持久化            │            │
│  │     · Binlog存档               │            │
│  └──────────────────────────────┘            │
│                                              │
└─────────────────────────────────────────────┘
```

**分片与副本策略**：
```
数据分布策略: Hash(collection_id + primary_key) % num_shards

┌────────────────────────────────────────────┐
│           分片与副本拓扑                     │
├────────────────────────────────────────────┤
│                                            │
│  分片0 [主] ──→ 分片0 [副] (DataNode 3)    │
│       │                                    │
│  分片1 [主] ──→ 分片1 [副] (DataNode 1)    │
│       │                                    │
│  分片2 [主] ──→ 分片2 [副] (DataNode 2)    │
│                                            │
│  读取策略：默认从主片读取（强一致性）        │
│          可配置为从副片读取（最终一致性）    │
│                                            │
│  写入策略：写入所有主副片后确认              │
│          可配置W=1提高吞吐（牺牲一致性）     │
│                                            │
└────────────────────────────────────────────┘
```

#### 1.4.6 Rerank服务

**概念**：Rerank服务使用交叉编码器（Cross-Encoder）模型对初步检索的候选文档进行精细排序。由于Cross-Encoder对每对(query, document)独立打分，计算复杂度为O(n)，通常只对Top-K候选集（如Top-100到Top-200）进行重排序。

**工作流程**：
```
检索阶段（第一阶段）:
  查询向量 → 向量数据库(HNSW/IVF) → Top-200 候选文档
           (毫秒级, 近似搜索)

重排序阶段（第二阶段）:
  Top-200 候选 → Cross-Encoder → 精排分数 → Top-10 最终结果
                  (百毫秒级, 精确打分)

┌──────────────────────────────────────────┐
│         Rerank 服务架构                   │
├──────────────────────────────────────────┤
│                                          │
│  请求 → [批聚合器] → [GPU推理池] → [排序]→│
│                                          │
│  批聚合器:                                │
│  · 合并多个查询的候选文档对               │
│  · 最大批大小: 128对                     │
│  · 最大等待: 20ms                        │
│                                          │
│  GPU推理池:                               │
│  · GPU 0: BGE-Reranker-v2-m3            │
│  · GPU 1: BGE-Reranker-v2-m3            │
│                                          │
│  模型对比:                                │
│  ┌────────────────────────────────────┐  │
│  │ 模型              │ CPU耗时 │ GPU耗时│  │
│  ├────────────────────────────────────┤  │
│  │ bge-reranker-base │ 15ms/对 │ 3ms/对│  │
│  │ bge-reranker-v2-m3│ 25ms/对 │ 5ms/对│  │
│  │ bge-reranker-large│ 30ms/对 │ 6ms/对│  │
│  └────────────────────────────────────┘  │
│                                          │
└──────────────────────────────────────────┘
```

#### 1.4.7 LLM推理服务

**概念**：LLM推理服务是RAG系统的"生成"环节。生产环境需要支持高并发请求、流式输出和高效的GPU利用率。vLLM和SGLang是当前主流的LLM推理引擎。

**推理引擎对比**：

| 特性 | vLLM | SGLang | TGI (HuggingFace) | TensorRT-LLM |
|------|------|--------|--------------------|---------------|
| PagedAttention | 支持 | 支持(原生) | 支持 | 支持 |
| 连续批处理 | 支持 | 支持 | 支持 | 支持 |
| 量化支持 | GPTQ/AWQ | GPTQ/AWQ | GPTQ/AWQ | 全面支持 |
| 前缀缓存 | 支持 | 支持(RadixAttention) | 不支持 | 支持 |
| 吞吐量(Qwen2-7B) | ~4000 tok/s | ~4500 tok/s | ~3000 tok/s | ~5000 tok/s |
| 部署复杂度 | 低 | 低 | 中 | 高 |
| 社区活跃度 | 极高 | 高 | 高 | 中 |

**vLLM部署配置**：
```yaml
# vLLM 服务部署配置
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
  namespace: rag-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-inference
  template:
    metadata:
      labels:
        app: llm-inference
    spec:
      nodeSelector:
        accelerator: nvidia-a100
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - "--model"
        - "Qwen/Qwen2-72B-Instruct-AWQ"
        - "--tensor-parallel-size"
        - "4"                # 跨4张GPU张量并行
        - "--max-model-len"
        - "32768"
        - "--gpu-memory-utilization"
        - "0.90"
        - "--max-num-seqs"
        - "128"              # 最大并发序列数
        - "--enable-prefix-caching"
        - "--enable-chunked-prefill"
        - "--max-num-batched-tokens"
        - "8192"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0,1,2,3"
        resources:
          limits:
            nvidia.com/gpu: 4
            memory: "256Gi"
          requests:
            nvidia.com/gpu: 4
            memory: "128Gi"
        ports:
        - containerPort: 8000
          name: http
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 10
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/huggingface
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: llm-inference
  namespace: rag-system
spec:
  type: ClusterIP
  selector:
    app: llm-inference
  ports:
  - port: 8000
    targetPort: 8000
```

#### 1.4.8 关系型数据库（PostgreSQL）

**概念**：关系型数据库在RAG系统中负责存储结构化元数据、用户信息、权限配置和审计日志。PostgreSQL因其对JSON类型的良好支持和丰富的扩展生态（pgvector），成为RAG系统的首选关系型数据库。

**数据模型设计**：
```sql
-- 核心表结构设计

-- 知识库表
CREATE TABLE knowledge_bases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    embedding_model VARCHAR(100) DEFAULT 'bge-m3',
    chunk_size INT DEFAULT 512,
    chunk_overlap INT DEFAULT 50,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(tenant_id, name)
);

-- 文档表
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kb_id UUID REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    filename VARCHAR(500) NOT NULL,
    file_type VARCHAR(50),
    file_size BIGINT,
    file_hash VARCHAR(64),
    storage_path VARCHAR(1000),
    status VARCHAR(50) DEFAULT 'uploaded',
    -- uploaded → parsing → chunking → embedding → indexed → ready
    chunk_count INT DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 文档块表
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    token_count INT,
    vector_id VARCHAR(255),  -- 向量数据库中对应的ID
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

-- 查询日志表（分区表，按月分区）
CREATE TABLE query_logs (
    id UUID DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    user_id UUID,
    query TEXT NOT NULL,
    retrieved_chunks JSONB,
    generated_answer TEXT,
    model_used VARCHAR(100),
    latency_ms INT,
    token_usage JSONB,
    cache_hit BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- 创建月度分区
CREATE TABLE query_logs_2026_06 PARTITION OF query_logs
    FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
CREATE TABLE query_logs_2026_07 PARTITION OF query_logs
    FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- 用户表（支持多租户）
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    username VARCHAR(100) NOT NULL,
    email VARCHAR(255),
    role VARCHAR(50) DEFAULT 'user',
    -- admin / manager / user / viewer
    quota_limit INT DEFAULT 1000,  -- 日查询限制
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(tenant_id, username)
);

-- 索引优化
CREATE INDEX idx_documents_kb_id ON documents(kb_id);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_query_logs_tenant_created ON query_logs(tenant_id, created_at DESC);
CREATE INDEX idx_query_logs_latency ON query_logs(latency_ms);
```

#### 1.4.9 消息队列（Apache Kafka）

**概念**：消息队列在RAG系统中承担异步处理和解耦的关键角色。文档上传后的解析、切片、向量化和入库是一个长流程，通过消息队列可以实现异步、可靠和可扩展的处理。

**Topic设计**：
```
Kafka Topic 拓扑:

┌─────────────────────────────────────────────┐
│              RAG系统 Kafka Topic设计          │
├─────────────────────────────────────────────┤
│                                              │
│  doc.uploaded (分区: 16, 副本: 2)            │
│  ├─ 生产者: API Gateway (上传接口)           │
│  └─ 消费者: 文档解析服务 × 4                 │
│     消息体: {doc_id, kb_id, tenant_id,       │
│              file_path, file_type, file_size} │
│                                              │
│  doc.parsed (分区: 16, 副本: 2)              │
│  ├─ 生产者: 文档解析服务                     │
│  └─ 消费者: 文档切片服务 × 4                  │
│     消息体: {doc_id, parsed_content,          │
│              parse_metadata, page_count}      │
│                                              │
│  chunks.created (分区: 32, 副本: 2)           │
│  ├─ 生产者: 文档切片服务                     │
│  └─ 消费者: Embedding服务 × 8                │
│     消息体: {doc_id, chunks: [{id, content,   │
│              chunk_index}], batch_id}         │
│                                              │
│  vectors.stored (分区: 16, 副本: 2)           │
│  ├─ 生产者: Embedding服务                    │
│  └─ 消费者: 索引更新服务 × 2                  │
│     消息体: {doc_id, vector_ids, status}      │
│                                              │
│  index.updated (分区: 8, 副本: 2)            │
│  ├─ 生产者: 索引更新服务                     │
│  └─ 消费者: 通知服务, 缓存失效服务            │
│     消息体: {kb_id, doc_count, timestamp}     │
│                                              │
│  query.logged (分区: 32, 副本: 1)             │
│  ├─ 生产者: 查询服务                         │
│  └─ 消费者: 分析服务, 成本追踪服务            │
│     消息体: {query_log完整记录}               │
│                                              │
└─────────────────────────────────────────────┘
```

#### 1.4.10 监控体系（Prometheus + Grafana + Jaeger + ELK）

**概念**：可观测性体系是保障RAG系统稳定运行的基石。通过指标（Metrics）、链路追踪（Tracing）和日志（Logging）三位一体，实现对系统全方位的监控。

**架构总览**：
```
┌──────────────────────────────────────────────────────┐
│               可观测性架构                             │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐         │
│  │  应用日志 │   │ 系统指标  │   │ 链路追踪  │         │
│  │  (Logs)  │   │ (Metrics) │   │ (Traces)  │         │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘         │
│       │              │              │                 │
│       ▼              ▼              ▼                 │
│  ┌─────────┐   ┌───────────┐   ┌─────────┐          │
│  │Filebeat │   │Prometheus │   │ Jaeger  │          │
│  │收集器   │   │  采集器    │   │ Agent   │          │
│  └────┬────┘   └─────┬─────┘   └────┬────┘          │
│       │              │              │                 │
│       ▼              ▼              ▼                 │
│  ┌─────────────────────────────────────────┐        │
│  │         Elasticsearch 集群                │        │
│  │  (日志存储 + APM数据 + 追踪数据)          │        │
│  └────────────────────┬────────────────────┘        │
│                       │                              │
│       ┌───────────────┼───────────────┐              │
│       │               │               │              │
│       ▼               ▼               ▼              │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐         │
│  │ Kibana  │   │ Grafana  │   │ Jaeger UI│         │
│  │ (日志)  │   │ (仪表盘) │   │ (追踪)   │         │
│  └─────────┘   └──────────┘   └──────────┘         │
│                                                      │
│  ┌──────────────────────────────────────────┐       │
│  │           AlertManager 告警管理            │       │
│  │  · PagerDuty / 钉钉 / 企业微信 / 邮件     │       │
│  └──────────────────────────────────────────┘       │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## 第二节：千万级文档RAG系统设计

### 2.1 规模挑战分析

当文档量从百万级跨越到千万级（1000万+文档），系统面临一系列新的挑战：

```
┌─────────────────────────────────────────────────────────────┐
│              百万级 vs 千万级 对比分析                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  维度           │ 百万级(1M)        │ 千万级(10M+)            │
│  ───────────────┼───────────────────┼────────────────────────│
│  文档数         │ 1,000,000         │ 10,000,000+            │
│  块(Chunk)数    │ 1亿               │ 10亿+                  │
│  向量存储       │ ~460 GB           │ ~4.6 TB+               │
│  日增文档       │ 1,000-5,000       │ 10,000-50,000          │
│  最大QPS        │ 1,000             │ 10,000+                │
│  向量检索        │ 毫秒级            │ 需要混合索引+分区       │
│  LLM并发        │ 50-100            │ 500-1,000              │
│  可用区         │ 单区域            │ 多区域                  │
│  数据一致性     │ 强一致            │ 最终一致(索引异步)       │
│  成本(月)       │ $20K-$50K         │ $150K-$500K            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 分片策略

#### 2.2.1 文档级分片

```
┌──────────────────────────────────────────────────────┐
│              文档级分片架构                            │
├──────────────────────────────────────────────────────┤
│                                                      │
│  分片键: hash(document_id) % num_shards               │
│                                                      │
│  ┌────────────────────────────────────────────┐      │
│  │             路由层 (Shard Router)            │      │
│  │  · 根据 tenant_id + kb_id 确定目标分片       │      │
│  │  · 支持分片再均衡 (Resharding)               │      │
│  │  · 元数据缓存: Redis 存储分片映射            │      │
│  └────────────────┬───────────────────────────┘      │
│                   │                                   │
│    ┌──────────────┼──────────────┐                    │
│    │              │              │                    │
│  ┌─▼──────┐  ┌───▼─────┐  ┌───▼─────┐              │
│  │ Shard 0│  │ Shard 1 │  │ Shard 2 │              │
│  │ doc    │  │ doc     │  │ doc     │              │
│  │ 0-3.3M │  │ 3.3-6.6M│  │ 6.6-10M │              │
│  │        │  │         │  │         │              │
│  │Milvus-0│  │Milvus-1 │  │Milvus-2 │              │
│  │ 向量库  │  │ 向量库   │  │ 向量库   │              │
│  │PG-0    │  │PG-1     │  │PG-2     │              │
│  │ 元数据  │  │ 元数据   │  │ 元数据   │              │
│  └────────┘  └─────────┘  └─────────┘              │
│                                                      │
│  跨分片查询: 查询广播到所有相关分片 → 结果合并 → 排序 │
│                                                      │
└──────────────────────────────────────────────────────┘
```

#### 2.2.2 集合级分片（按知识库/租户）

```
按知识库隔离的向量存储策略:

租户A:                            租户B:
┌──────────┐                     ┌──────────┐
│ KB-A-1   │                     │ KB-B-1   │
│ Collection│                    │ Collection│
│ 100万文档 │                    │ 200万文档 │
└──────────┘                     └──────────┘
┌──────────┐                     ┌──────────┐
│ KB-A-2   │                     │ KB-B-2   │
│ Collection│                    │ Collection│
│ 50万文档  │                    │ 300万文档 │
└──────────┘                     └──────────┘

优势:
· 租户间物理隔离，安全性更高
· 每个Collection独立索引，检索性能不受其他租户影响
· 可独立扩容热点知识库

劣势:
· 跨知识库查询需要多次检索+结果合并
· 小知识库资源浪费
· 管理复杂度增加
```

### 2.3 多区域部署架构

```
┌─────────────────────────────────────────────────────────────┐
│                千万级RAG多区域部署架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                         ┌──────────┐                        │
│                         │ 全局DNS   │                        │
│                         │ Geo-Route│                        │
│                         └────┬─────┘                        │
│                              │                              │
│        ┌─────────────────────┼─────────────────────┐        │
│        │                     │                     │        │
│   ┌────▼─────┐          ┌────▼─────┐          ┌───▼──────┐ │
│   │  Region A │          │  Region B │          │ Region C │ │
│   │  (主区域) │          │  (从区域) │          │ (从区域) │ │
│   │           │          │           │          │          │ │
│   │ ┌───────┐ │  同步    │ ┌───────┐ │  同步    │ ┌──────┐ │ │
│   │ │ 完整  │ │◄────────┤ │ 完整  │ │◄────────┤ │完整  │ │ │
│   │ │ 服务栈│ │  CDC    │ │ 服务栈│ │  CDC    │ │服务栈│ │ │
│   │ └───────┘ │          │ └───────┘ │          │ └──────┘ │ │
│   │           │          │           │          │          │ │
│   │ · API GW  │          │ · API GW  │          │· API GW  │ │
│   │ · Embed   │          │ · Embed   │          │· Embed   │ │
│   │ · Milvus  │          │ · Milvus  │          │· Milvus  │ │
│   │ · Rerank  │          │ · Rerank  │          │· Rerank  │ │
│   │ · LLM     │          │ · LLM     │          │· LLM     │ │
│   │ · PG(主)  │          │ · PG(只读)│          │· PG(只读)│ │
│   └─────┬─────┘          └─────┬─────┘          └────┬────┘ │
│         │                      │                     │       │
│         └──────────────────────┼─────────────────────┘       │
│                                │                             │
│                     ┌──────────▼──────────┐                  │
│                     │  全局消息队列        │                  │
│                     │  Kafka (跨区域复制)  │                  │
│                     └─────────────────────┘                  │
│                                                             │
│  数据同步策略:                                               │
│  · 主区域写入 → CDC (Change Data Capture) → 从区域异步复制   │
│  · 向量索引: 每日全量同步 + 增量实时同步                     │
│  · 文档文件: S3跨区域复制 (CRR)                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 冷热数据分层

```
┌──────────────────────────────────────────────────────┐
│             冷热数据分层策略                           │
├──────────────────────────────────────────────────────┤
│                                                      │
│  热数据层 (Hot Tier) - NVMe SSD                        │
│  ┌────────────────────────────────────────────┐      │
│  │ · 近30天更新的知识库                         │      │
│  │ · 高频查询的向量索引 (Top 20% 知识库)        │      │
│  │ · 活跃租户数据                              │      │
│  │ · 存储: 20%的数据, 承担80%的查询             │      │
│  │ · 延迟: <10ms P99                           │      │
│  └────────────────────────────────────────────┘      │
│                                                      │
│  温数据层 (Warm Tier) - SATA SSD / 高性能HDD          │
│  ┌────────────────────────────────────────────┐      │
│  │ · 30-90天内更新的知识库                      │      │
│  │ · 中频查询数据                              │      │
│  │ · 存储: 30%的数据                           │      │
│  │ · 延迟: <50ms P99                           │      │
│  └────────────────────────────────────────────┘      │
│                                                      │
│  冷数据层 (Cold Tier) - 对象存储 (S3/MinIO)           │
│  ┌────────────────────────────────────────────┐      │
│  │ · 90天以上未更新的知识库                     │      │
│  │ · 低频查询数据 (归档)                       │      │
│  │ · 存储: 50%的数据                           │      │
│  │ · 延迟: <500ms P99 (需预热后查询)           │      │
│  │ · 使用DiskANN/Mmap索引降低内存开销           │      │
│  └────────────────────────────────────────────┘      │
│                                                      │
│  自动分层策略:                                        │
│  · 基于 LRU (Least Recently Used) + 时间衰减         │
│  · 每日定时任务评估数据热度和迁移                    │
│  · 冷数据查询时自动预热到温层                        │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 2.5 千万级性能基准与SLA

| SLA指标 | 百万级目标 | 千万级目标 | 实现策略 |
|---------|-----------|-----------|---------|
| 查询延迟 P50 | <1s | <1.5s | 多级缓存+并行检索 |
| 查询延迟 P95 | <2s | <3s | 超时降级+部分返回 |
| 查询延迟 P99 | <5s | <8s | 异步队列+监控告警 |
| 系统可用性 | 99.9% | 99.95% | 多区域+自动故障转移 |
| 数据持久性 | 99.999% | 99.9999% | 跨区域复制+版本快照 |
| 文档摄入吞吐 | 1000 docs/min | 5000 docs/min | 水平扩展Kafka消费者 |
| 向量检索QPS | 5,000 | 50,000 | 分片+副本扩展 |
| LLM并发数 | 100 | 500 | vLLM集群水平扩展 |
| RPO (恢复点目标) | <5分钟 | <1分钟 | CDC实时同步 |
| RTO (恢复时间目标) | <30分钟 | <10分钟 | 自动故障转移+K8s自愈 |

### 2.6 灾备与恢复架构

```
┌──────────────────────────────────────────────────────┐
│               灾备架构设计                             │
├──────────────────────────────────────────────────────┤
│                                                      │
│  备份策略:                                            │
│  ┌────────────────────────────────────────────┐      │
│  │  组件         │ 备份方式        │ 频率      │      │
│  ├────────────────────────────────────────────┤      │
│  │  PostgreSQL   │ pg_dump + WAL  │ 持续+每日 │      │
│  │  向量数据库    │ 快照+Binlog    │ 每日+持续 │      │
│  │  对象存储      │ 跨区域复制(CRR)│ 实时      │      │
│  │  Redis        │ RDB + AOF      │ 每小时    │      │
│  │  Kafka        │ MirrorMaker    │ 实时      │      │
│  │  K8s配置      │ Git + ArgoCD   │ 变更时    │      │
│  └────────────────────────────────────────────┘      │
│                                                      │
│  恢复流程:                                            │
│  1. 检测故障 → 2. 触发告警 → 3. DNS切换               │
│  4. 备用区域接管 → 5. 数据校验 → 6. 服务恢复           │
│                                                      │
│  演练要求:                                            │
│  · 每季度一次全量灾备演练                             │
│  · 每月一次关键组件恢复演练                           │
│  · 自动化恢复时间 < 15分钟                            │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## 第三节：部署架构

### 3.1 Kubernetes原生化部署

#### 3.1.1 整体部署拓扑

```
┌─────────────────────────────────────────────────────────────┐
│              Kubernetes 集群部署拓扑                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Ingress Controller                  │   │
│  │              (Nginx Ingress / Traefik)                │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │                    Service Mesh                       │   │
│  │               (Istio / Linkerd)                       │   │
│  │                                                      │   │
│  │  · 东西向流量管理 (mTLS, 负载均衡, 熔断)              │   │
│  │  · 南北向流量管理 (Ingress Gateway)                   │   │
│  │  · 可观测性 (分布式追踪, 指标采集)                     │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │                  微服务层                              │   │
│  │                                                      │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │   │
│  │  │API GW  │ │Query   │ │Admin   │ │Async   │        │   │
│  │  │(×3)    │ │Svc (×5)│ │Svc (×3)│ │Worker  │        │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘        │   │
│  │                                                      │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │   │
│  │  │Embed   │ │Rerank  │ │LLM     │ │Vector  │        │   │
│  │  │Svc (×4)│ │Svc (×2)│ │Svc (×4)│ │DB Op   │        │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘        │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │                  中间件层                              │   │
│  │                                                      │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │   │
│  │  │Redis   │ │Kafka   │ │PostgreSQL│ │MinIO  │        │   │
│  │  │Cluster │ │Cluster │ │(主+只读)│ │Cluster│        │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘        │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │               GPU节点池 (专用节点)                     │   │
│  │                                                      │   │
│  │  ┌──────────────────────────────────────────────┐    │   │
│  │  │  Embedding Pool: 4×A100-80G                  │    │   │
│  │  │  Rerank Pool:    2×A10-24G                   │    │   │
│  │  │  LLM Pool:       8×A100-80G (4卡×2节点)      │    │   │
│  │  └──────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 3.1.2 Helm Chart 部署

```yaml
# values.yaml - RAG系统Helm Chart值文件
global:
  namespace: rag-system
  imageRegistry: registry.example.com/rag
  imagePullSecrets:
    - name: registry-credentials
  environment: production

# API网关
apiGateway:
  enabled: true
  replicas: 3
  image: apisix:3.8.0
  resources:
    requests:
      cpu: "2"
      memory: "4Gi"
    limits:
      cpu: "4"
      memory: "8Gi"
  service:
    type: ClusterIP
    port: 9080
  config:
    jwtSecret: ""    # 从外部Secret注入
    rateLimit:
      default: "100/min"
      premium: "500/min"

# 查询服务
queryService:
  replicas: 5
  image: rag-query-service:v2.3.0
  resources:
    requests:
      cpu: "2"
      memory: "4Gi"
    limits:
      cpu: "4"
      memory: "8Gi"
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 20
    targetCPUUtilization: 70
    targetMemoryUtilization: 80

# Embedding服务 (GPU)
embeddingService:
  replicas: 4
  image: rag-embedding-service:v2.3.0
  modelName: "BAAI/bge-m3"
  maxBatchSize: 256
  resources:
    requests:
      cpu: "4"
      memory: "16Gi"
      nvidia.com/gpu: 1
    limits:
      cpu: "8"
      memory: "32Gi"
      nvidia.com/gpu: 1
  nodeSelector:
    accelerator: nvidia-a100
  tolerations:
    - key: "nvidia.com/gpu"
      operator: "Exists"
      effect: "NoSchedule"

# LLM推理服务 (vLLM)
llmService:
  replicas: 2  # 每副本4 GPU
  image: vllm/vllm-openai:v0.4.3
  modelName: "Qwen/Qwen2-72B-Instruct-AWQ"
  tensorParallelSize: 4
  maxModelLen: 32768
  maxNumSeqs: 128
  gpuMemoryUtilization: 0.90
  resources:
    requests:
      nvidia.com/gpu: 4
      memory: "256Gi"
    limits:
      nvidia.com/gpu: 4
      memory: "256Gi"
  nodeSelector:
    accelerator: nvidia-a100-80gb
  autoscaling:
    enabled: true
    minReplicas: 1
    maxReplicas: 5
    targetMetric: nvidia_gpu_utilization
    targetValue: 75

# 向量数据库 (Milvus)
milvus:
  mode: cluster
  proxy:
    replicas: 3
  queryNode:
    replicas: 3
  dataNode:
    replicas: 3
  indexNode:
    replicas: 2
  persistence:
    enabled: true
    storageClass: "ssd-sc"
    size: "2Ti"
  etcd:
    replicas: 3
  minio:
    mode: distributed
    replicas: 4
    persistence:
      size: "10Ti"

# Redis缓存
redis:
  mode: cluster
  replicas: 6
  password: ""  # 从外部Secret注入
  persistence:
    enabled: true
    size: "100Gi"
  config:
    maxmemory: "16gb"
    maxmemory-policy: "allkeys-lru"

# PostgreSQL
postgresql:
  replicas: 1  # 主库
  readReplicas: 2  # 只读副本
  persistence:
    size: "500Gi"
  backup:
    enabled: true
    schedule: "0 2 * * *"  # 每日凌晨2点
    retentionDays: 30

# Kafka
kafka:
  replicas: 3
  persistence:
    size: "1Ti"
  topics:
    - name: doc.uploaded
      partitions: 16
      replicationFactor: 2
    - name: chunks.created
      partitions: 32
      replicationFactor: 2
    - name: query.logged
      partitions: 32
      replicationFactor: 2

# 监控
monitoring:
  prometheus:
    enabled: true
    retention: "30d"
  grafana:
    enabled: true
    adminPassword: ""  # Secret注入
    dashboards:
      - rag-overview
      - rag-llm-metrics
      - rag-vector-db
      - rag-business-metrics
  jaeger:
    enabled: true
    storage:
      type: elasticsearch
  alertManager:
    enabled: true
    config:
      receivers:
        - name: "pagerduty"
        - name: "dingtalk"
        - name: "email-ops"
```

#### 3.1.3 自动扩缩容配置

```yaml
# HPA (Horizontal Pod Autoscaler) 配置
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: query-service-hpa
  namespace: rag-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: query-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: rag_query_qps
      target:
        type: AverageValue
        averageValue: "100"   # 每个Pod承载100 QPS
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # 5分钟稳定窗口
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
      - type: Pods
        value: 5
        periodSeconds: 60
      selectPolicy: Max
---
# GPU服务HPA (基于GPU利用率)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: embedding-service-hpa
  namespace: rag-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: embedding-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: DCGM_FI_DEV_GPU_UTIL
      target:
        type: AverageValue
        averageValue: "75"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 600  # GPU服务缩容更保守
```

### 3.2 CI/CD流水线

```
┌──────────────────────────────────────────────────────────┐
│              RAG系统 CI/CD 流水线                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  代码提交 ──→ ┌────────────┐                             │
│              │  代码检查    │                            │
│              │ · Lint      │                            │
│              │ · SAST      │                            │
│              │ · 单元测试   │                            │
│              └─────┬──────┘                             │
│                    │                                     │
│              ┌─────▼──────┐                             │
│              │  构建镜像    │                            │
│              │ · Docker    │                            │
│              │ · 安全扫描   │                            │
│              │ · 签名      │                            │
│              └─────┬──────┘                             │
│                    │                                     │
│              ┌─────▼──────┐                             │
│              │  集成测试    │                            │
│              │ · API测试    │                            │
│              │ · 检索质量    │                            │
│              │ · 性能基准    │                            │
│              └─────┬──────┘                             │
│                    │                                     │
│         ┌──────────┼──────────┐                         │
│         │          │          │                         │
│    ┌────▼───┐ ┌───▼────┐ ┌──▼──────┐                   │
│    │ 蓝绿部署 │ │金丝雀  │ │滚动更新 │                   │
│    │ (索引)  │ │ (模型) │ │ (API)   │                   │
│    └────────┘ └────────┘ └─────────┘                   │
│                                                          │
│  索引更新策略（蓝绿部署）：                                │
│  1. 在新环境构建完整索引                                  │
│  2. 索引质量验证（覆盖率、延迟基准）                      │
│  3. 流量切换到新索引                                      │
│  4. 旧索引保留24小时作为回滚点                            │
│                                                          │
│  模型更新策略（金丝雀部署）：                              │
│  1. 新模型部署到5%流量                                    │
│  2. 监控质量指标（准确率、Token使用）                     │
│  3. 逐步扩大到25% → 50% → 100%                           │
│  4. 任何异常自动回滚                                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**GitHub Actions CI/CD配置**：
```yaml
# .github/workflows/deploy-rag.yml
name: Deploy RAG System

on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'config/**'
      - 'helm/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Unit Tests
        run: |
          pip install -r requirements-dev.txt
          pytest tests/ --cov=src --cov-report=xml
      - name: Integration Tests
        run: |
          docker-compose -f docker-compose.test.yml up -d
          pytest tests/integration/
      
  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker Images
        run: |
          docker build -t $REGISTRY/query-service:$GITHUB_SHA .
          docker push $REGISTRY/query-service:$GITHUB_SHA
      - name: Security Scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/query-service:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
  
  deploy-canary:
    needs: build-and-push
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy Canary
        run: |
          helm upgrade rag-system ./helm/rag-system \
            --set queryService.image.tag=$GITHUB_SHA \
            --set canary.enabled=true \
            --set canary.weight=5 \
            --namespace rag-system \
            --wait
      
  promote:
    needs: deploy-canary
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Verify Canary Metrics
        run: |
          sleep 600  # 等待10分钟观察
          ./scripts/verify-canary.sh
      - name: Promote to Stable
        run: |
          helm upgrade rag-system ./helm/rag-system \
            --set queryService.image.tag=$GITHUB_SHA \
            --set canary.enabled=false \
            --namespace rag-system \
            --wait
```

---

## 第四节：监控与可观测性

### 4.1 关键指标体系

#### 4.1.1 指标分类与采集

```
┌─────────────────────────────────────────────────────────────┐
│              RAG系统 监控指标全景                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  一、检索性能指标 (Retrieval Metrics)                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 指标名                 │ 类型    │ 标签               │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ rag_retrieval_latency  │ Histogram│ model, kb_id     │   │
│  │ rag_retrieval_qps      │ Gauge   │ service           │   │
│  │ rag_retrieval_topk_hit │ Counter │ kb_id, query_type │   │
│  │ rag_rerank_latency     │ Histogram│ model             │   │
│  │ rag_embedding_latency  │ Histogram│ model, batch_size │   │
│  │ rag_vector_db_qps      │ Gauge   │ collection        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  二、LLM服务指标 (LLM Metrics)                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 指标名                 │ 类型    │ 标签               │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ llm_request_latency    │ Histogram│ model             │   │
│  │ llm_token_usage_total  │ Counter │ model, type       │   │
│  │ llm_throughput_tokens  │ Gauge   │ model             │   │
│  │ llm_queue_length       │ Gauge   │ model             │   │
│  │ llm_gpu_utilization    │ Gauge   │ gpu_id, model     │   │
│  │ llm_active_requests    │ Gauge   │ model             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  三、缓存指标 (Cache Metrics)                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ rag_cache_hit_ratio     │ Gauge   │ cache_level      │   │
│  │ rag_cache_hits_total    │ Counter │ cache_level      │   │
│  │ rag_cache_misses_total  │ Counter │ cache_level      │   │
│  │ rag_cache_size_bytes    │ Gauge   │ cache_level      │   │
│  │ rag_cache_evictions     │ Counter │ cache_level      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  四、业务指标 (Business Metrics)                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ rag_query_total         │ Counter │ tenant_id, status│   │
│  │ rag_query_success_rate  │ Gauge   │ tenant_id        │   │
│  │ rag_doc_ingested_total  │ Counter │ tenant_id, type  │   │
│  │ rag_cost_per_query      │ Histogram│ tenant_id       │   │
│  │ rag_active_users        │ Gauge   │ tenant_id        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  五、基础设施指标 (Infrastructure Metrics)                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ node_cpu_utilization    │ Gauge   │ node             │   │
│  │ node_memory_utilization │ Gauge   │ node             │   │
│  │ pod_restart_count       │ Counter │ pod              │   │
│  │ kafka_consumer_lag      │ Gauge   │ topic, group     │   │
│  │ pg_connection_count     │ Gauge   │ instance         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 4.1.2 指标采集配置

```yaml
# ServiceMonitor 配置 (用于 Prometheus Operator)
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: rag-query-service
  namespace: rag-system
spec:
  selector:
    matchLabels:
      app: query-service
  endpoints:
  - port: metrics
    interval: 15s
    path: /metrics
    relabelings:
    - sourceLabels: [__meta_kubernetes_pod_name]
      targetLabel: pod
    - sourceLabels: [__meta_kubernetes_namespace]
      targetLabel: namespace
---
# Prometheus 告警规则
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: rag-alerting-rules
  namespace: rag-system
spec:
  groups:
  - name: rag-performance
    rules:
    # 检索延迟告警
    - alert: HighRetrievalLatency
      expr: |
        histogram_quantile(0.95, 
          rate(rag_retrieval_latency_bucket[5m])) > 3
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "检索P95延迟超过3秒"
        description: "知识库 {{ $labels.kb_id }} P95检索延迟: {{ $value }}s"
    
    # LLM延迟告警
    - alert: HighLLMLatency
      expr: |
        histogram_quantile(0.95,
          rate(llm_request_latency_bucket[5m])) > 10
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "LLM P95延迟超过10秒"
    
    # 缓存命中率告警
    - alert: LowCacheHitRate
      expr: |
        rate(rag_cache_hits_total[5m]) /
        (rate(rag_cache_hits_total[5m]) + 
         rate(rag_cache_misses_total[5m])) < 0.3
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "缓存命中率低于30%"
    
    # GPU利用率异常
    - alert: GPUUtilizationLow
      expr: |
        DCGM_FI_DEV_GPU_UTIL < 20
      for: 30m
      labels:
        severity: info
      annotations:
        summary: "GPU利用率低于20%，可能存在资源浪费"
    
    # 错误率告警
    - alert: HighErrorRate
      expr: |
        rate(rag_query_errors_total[5m]) /
        rate(rag_query_total[5m]) > 0.05
      for: 3m
      labels:
        severity: critical
      annotations:
        summary: "查询错误率超过5%"
    
    # 向量数据库内存压力
    - alert: VectorDBMemoryPressure
      expr: |
        milvus_proxy_memory_usage_bytes /
        milvus_proxy_memory_limit_bytes > 0.85
      for: 10m
      labels:
        severity: warning
    
    # Kafka消费积压
    - alert: KafkaConsumerLag
      expr: |
        kafka_consumer_group_lag > 10000
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Kafka消费组 {{ $labels.group }} 积压 {{ $value }}"
    
    # 服务可用性
    - alert: ServiceDown
      expr: |
        up{job=~"rag-.*"} == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "RAG服务 {{ $labels.job }} 不可用"
```

### 4.2 Grafana仪表盘

#### 4.2.1 仪表盘设计

```
┌──────────────────────────────────────────────────────┐
│         Grafana仪表盘 - RAG系统总览                    │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────────────┐ ┌──────────────────────┐  │
│  │  QPS (实时)          │ │  P95 延迟 (实时)      │  │
│  │  ┌────────────────┐  │ │  ┌────────────────┐  │  │
│  │  │ 当前: 856/s    │  │ │  │ 检索: 1.2s     │  │  │
│  │  │ ▲ 12% vs 1h    │  │ │  │ LLM:  3.8s     │  │  │
│  │  │ ████████████   │  │ │  │ 端到端: 4.5s   │  │  │
│  │  └────────────────┘  │ │  └────────────────┘  │  │
│  └──────────────────────┘ └──────────────────────┘  │
│                                                      │
│  ┌──────────────────────┐ ┌──────────────────────┐  │
│  │  缓存命中率           │ │  错误率               │  │
│  │  ┌────────────────┐  │ │  ┌────────────────┐  │  │
│  │  │ L1: 12%        │  │ │  │ 当前: 0.23%    │  │  │
│  │  │ L2: 45%        │  │ │  │ 阈值: 1%       │  │  │
│  │  │ L3: 18%        │  │ │  │ ▂▂▂▂▂▂▂▂▁▁   │  │  │
│  │  └────────────────┘  │ │  └────────────────┘  │  │
│  └──────────────────────┘ └──────────────────────┘  │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │  LLM Token 使用趋势 (24h)                      │    │
│  │  ▁▁▂▂▃▃▄▅▅▆▆▇▇▇█████████████░░░░            │    │
│  │  Input: 12.5M tokens  │  Output: 3.2M tokens  │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │  GPU 利用率 (实时)                              │    │
│  │  GPU0: [████████░░] 85%                       │    │
│  │  GPU1: [█████████░] 92%                       │    │
│  │  GPU2: [███████░░░] 74%                       │    │
│  │  GPU3: [████████░░] 81%                       │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │  检索延迟分布 (Histogram)                      │    │
│  │  P50: 0.5s  P90: 2.1s  P95: 2.8s  P99: 4.2s│    │
│  │  ┌────────────────────────────────────────┐   │    │
│  │  │ ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│   │    │
│  │  │ 0s   1s   2s   3s   4s   5s            │   │    │
│  │  └────────────────────────────────────────┘   │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 4.3 SLA定义与成本追踪

| SLA级别 | 可用性 | 延迟(P95) | 适用场景 | 赔偿 |
|---------|--------|-----------|---------|------|
| 企业版 | 99.99% | <2s 简单 / <5s 复杂 | 金融/医疗 | 违约赔偿 30% |
| 专业版 | 99.9% | <3s 简单 / <8s 复杂 | 一般企业 | 违约赔偿 10% |
| 标准版 | 99.5% | <5s 简单 / <15s 复杂 | 小型团队 | 服务积分 |

**成本追踪模型**：
```
单次查询成本 = 检索成本 + 重排序成本 + LLM成本 + 基础设施成本

检索成本   = 向量数据库计算时间 × 计算单价
           ≈ 0.01s × $0.02/s ≈ $0.0002/query

重排序成本 = GPU计算时间 × GPU单价
           ≈ 0.03s × $0.15/s ≈ $0.0045/query

LLM成本    = 输入Token × 输入单价 + 输出Token × 输出单价
           ≈ 2000 × $0.0005/K + 500 × $0.002/K
           ≈ $0.001 + $0.001 = $0.002/query

基础设施成本 = (服务器+存储+网络) / 总查询数
             ≈ $500/天 / 100万查询/天 ≈ $0.0005/query

────────
单次查询总成本 ≈ $0.0072/query

月成本 ≈ $0.0072 × 3000万查询/月 ≈ $216,000/月 (千万级规模)
```

---

## 第五节：安全架构

### 5.1 多租户数据隔离

```
┌──────────────────────────────────────────────────────┐
│              多租户安全架构                            │
├──────────────────────────────────────────────────────┤
│                                                      │
│  隔离级别选择:                                        │
│  ┌────────────────────────────────────────────┐      │
│  │  级别      │ 方式          │ 隔离度 │ 成本  │      │
│  ├────────────────────────────────────────────┤      │
│  │  物理隔离  │ 独立集群      │ ★★★★★ │ 极高  │      │
│  │  逻辑隔离  │ 共享集群+分区  │ ★★★★☆ │ 中等  │      │
│  │  Collection│ 独立向量集合  │ ★★★☆☆ │ 较低  │      │
│  │  字段级    │ tenant_id过滤 │ ★★☆☆☆ │ 最低  │      │
│  └────────────────────────────────────────────┘      │
│                                                      │
│  推荐方案（平衡安全与成本）：                           │
│  · 大型租户（千万级文档）：独立Collection+专用索引     │
│  · 中型租户（百万级文档）：独立Collection              │
│  · 小型租户（万级以下）：共享Collection+tenant_id过滤  │
│                                                      │
│  租户上下文贯穿全链路：                                │
│  API Gateway → 查询服务 → 检索 → LLM → 日志           │
│  每层验证 tenant_id 并附加到日志/追踪上下文            │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 5.2 API安全

```yaml
# API认证与授权配置
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: rag-api-authz
  namespace: rag-system
spec:
  selector:
    matchLabels:
      app: api-gateway
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/rag-system/sa/api-gateway"]
    to:
    - operation:
        methods: ["GET", "POST"]
        paths: ["/api/v1/query", "/api/v1/search"]
    when:
    - key: request.auth.claims[role]
      values: ["admin", "manager", "user", "viewer"]
  - from:
    - source:
        principals: ["cluster.local/ns/rag-system/sa/api-gateway"]
    to:
    - operation:
        methods: ["POST"]
        paths: ["/api/v1/upload", "/api/v1/admin/*"]
    when:
    - key: request.auth.claims[role]
      values: ["admin", "manager"]
---
# 网络策略 - 零信任网络
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: rag-zero-trust
  namespace: rag-system
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: istio-system
    ports:
    - port: 8080
      protocol: TCP
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis-cluster
    ports:
    - port: 6379
      protocol: TCP
  - to:
    - podSelector:
        matchLabels:
          app: postgresql
    ports:
    - port: 5432
      protocol: TCP
```

### 5.3 审计日志

```python
# 审计日志模型
class AuditLogger:
    """RAG系统审计日志记录器"""
    
    AUDIT_EVENTS = {
        "DOCUMENT_UPLOAD": "文档上传",
        "DOCUMENT_DELETE": "文档删除",
        "QUERY_EXECUTED": "查询执行",
        "KB_CREATED": "知识库创建",
        "KB_DELETED": "知识库删除",
        "USER_CREATED": "用户创建",
        "USER_DELETED": "用户删除",
        "PERMISSION_CHANGED": "权限变更",
        "API_KEY_CREATED": "API密钥创建",
        "MODEL_CONFIG_CHANGED": "模型配置变更",
    }
    
    async def log_event(self, 
                        event_type: str,
                        tenant_id: str,
                        user_id: str,
                        resource_id: str = None,
                        details: dict = None,
                        ip_address: str = None,
                        user_agent: str = None):
        """记录审计事件"""
        audit_record = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "event_name": self.AUDIT_EVENTS.get(event_type, event_type),
            "tenant_id": tenant_id,
            "user_id": user_id,
            "resource_id": resource_id,
            "details": details or {},
            "ip_address": ip_address,
            "user_agent": user_agent,
            "timestamp": datetime.utcnow().isoformat(),
            "source_service": os.getenv("SERVICE_NAME", "unknown"),
        }
        
        # 写入审计日志表 + Kafka Topic
        await self.db.insert("audit_logs", audit_record)
        await self.kafka.send("audit.events", audit_record)
    
    async def has_pii(self, content: str) -> bool:
        """检测PII内容"""
        patterns = {
            "phone": r'\b1[3-9]\d{9}\b',
            "id_card": r'\b\d{17}[\dXx]\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "bank_card": r'\b\d{16,19}\b',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        }
        for pii_type, pattern in patterns.items():
            if re.search(pattern, content):
                return True
        return False
```

---

## 第六节：工程实践与面试要点

### 6.1 企业最佳实践清单

```
┌─────────────────────────────────────────────────────────────┐
│              RAG系统 企业级最佳实践                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  架构设计:                                                   │
│  □ 采用微服务架构，各组件独立部署和扩缩容                     │
│  □ 所有服务无状态设计，状态外置到数据库/缓存                  │
│  □ 异步处理文档入库流程，Kafka解耦生产与消费                  │
│  □ 多级缓存策略，L1-L4逐级降低延迟和成本                     │
│  □ 读写分离（PostgreSQL主库写入，只读副本查询）               │
│                                                             │
│  性能优化:                                                   │
│  □ Embedding批量处理（最小批次32，理想批次128-256）           │
│  □ 向量数据库使用IVF+PQ量化降低内存占用                       │
│  □ LLM使用AWQ/GPTQ量化减少GPU显存需求                       │
│  □ 使用vLLM前缀缓存优化RAG场景（相似System Prompt）           │
│  □ 连接池优化：数据库连接数 = (2 * CPU核数) + 磁盘数          │
│                                                             │
│  可靠性:                                                     │
│  □ 所有关键组件至少3副本部署                                  │
│  □ 跨可用区部署（至少2个AZ）                                  │
│  □ 定期灾备演练（每季度全量，每月关键组件）                    │
│  □ 优雅降级：向量库不可用时返回关键词搜索结果                  │
│  □ 超时策略：检索3s + 重排2s + LLM 15s = 总20s               │
│                                                             │
│  安全合规:                                                   │
│  □ API层全量JWT认证+RBAC权限控制                             │
│  □ 服务间mTLS通信加密                                        │
│  □ 敏感数据脱敏（PII检测+自动掩码）                           │
│  □ 审计日志至少保留1年                                       │
│  □ 遵循数据本地化法规（中国《数据安全法》、GDPR）              │
│                                                             │
│  DevOps:                                                     │
│  □ 基础设施即代码（Terraform/Pulumi）                        │
│  □ 配置管理（Helm + ArgoCD GitOps）                          │
│  □ 监控先行：先部署监控，再部署业务                            │
│  □ 变更管理：所有基础设施变更走代码审查流程                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 常见面试问题

**Q1：百万级和千万级RAG系统在架构上的核心区别是什么？**

> 核心区别在于分片策略、数据一致性和成本优化。百万级可以采用单一向量数据库集群，千万级必须分片；百万级可以保证强一致性，千万级需要权衡最终一致性；千万级必须引入冷热分层以控制成本。

**Q2：RAG系统中如何设计缓存策略？**

> 采用L1-L4四级缓存：L1结果缓存（热点问题直接返回），L2向量缓存（避免重复Embedding），L3语义缓存（相似查询复用），L4 Token缓存（相同Prompt避免重复LLM计算）。缓存命中率是降低延迟和成本的核心杠杆。

**Q3：为什么选择Milvus而不是Elasticsearch做向量检索？**

> Milvus专为向量检索设计，原生支持GPU加速和分布式架构，十亿级向量的检索延迟远低于ES。ES的向量功能是后期添加的，在大规模场景下性能和成本均不占优。但ES在全文检索和日志分析场景仍有优势。实际生产中常采用Milvus（向量）+ ES（关键词）混合检索。

**Q4：RAG系统的SLA如何定义？**

> 核心SLA指标：可用性99.9%+，P95延迟<2s（简单查询）/ <5s（复杂查询），数据持久性99.999%+。需要明确测量间隔（如5分钟滑动窗口）和排除计划内维护时间。SLA应分层定义（检索、生成、端到端）。

**Q5：多租户RAG系统如何保证数据隔离？**

> 分层隔离策略：大租户使用独立Collection/库，中租户共享集群但独立命名空间，小租户使用tenant_id字段过滤。租户上下文通过TraceID在全链路传递，每层服务独立验证租户权限。向量检索时必须附加tenant_id过滤条件。

**Q6：如何处理RAG系统中的"冷启动"问题？**

> 冷启动包括：新知识库无查询数据（使用默认检索参数+AB测试调优）、新模型上线（金丝雀部署+自动化评估）、灾备切换后缓存为空（预热脚本+渐进式放量）、GPU冷启动（模型预加载+Keep-Alive探针）。

---

## 本章总结

企业级RAG系统的架构设计需要在性能、成本、可靠性和安全性之间寻找平衡。关键要点：

| 维度 | 核心决策 | 权衡考量 |
|------|---------|---------|
| 规模 | 分片策略（文档级 vs 集合级） | 复杂度 vs 隔离性 |
| 部署 | Kubernetes + Helm + GitOps | 灵活性 vs 学习曲线 |
| 缓存 | L1-L4多级缓存 | 命中率 vs 一致性 |
| 检索 | 向量（语义）+ 关键词（精确）混合 | 召回率 vs 精度 |
| LLM | vLLM连续批处理 + 量化 | 吞吐 vs 质量 |
| 监控 | 指标+追踪+日志三位一体 | 成本 vs 可观测性 |
| 安全 | 分层隔离 + 零信任网络 | 安全 vs 运维复杂度 |

企业应根据自身文档规模、查询负载、预算约束和安全合规要求，选择合适的技术架构。建议从百万级起步，在中型规模验证架构合理性后，逐步向千万级演进，避免过度设计。

---

*本章由AI架构师编写，结合了分布式系统、信息检索和大语言模型服务的最新工程实践。*
