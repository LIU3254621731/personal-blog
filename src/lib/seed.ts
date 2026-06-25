/**
 * Seed data for in-memory fallback (EdgeOne / environments without SQLite).
 *
 * This populates the in-memory stores with representative content so the
 * deployed site renders meaningful pages instead of empty shells.
 * Full CRUD + persistence requires CloudBase (see DEPLOY.md).
 */

import type { Post, Project, GardenEntry } from "./db";

function now(): string {
  return new Date().toISOString();
}

// ─── Seed Posts (last 6 published articles) ───────────────

export const SEED_POSTS: Array<Omit<Post, "createdAt" | "updatedAt"> & { createdAt: string; updatedAt: string }> = [
  {
    id: "seed_post_1", title: "从 RAG 到 Agentic RAG：AI 知识检索的进化之路",
    slug: "from-rag-to-agentic-rag",
    content: `## RAG 基础回顾\n\n检索增强生成已成为 LLM 应用的核心架构。\n\n### Agentic RAG 的突破\n\n- 多步推理：Agent 拆解复杂问题为子任务\n- 工具调用：按需调用搜索、计算等工具\n- 自我纠错：验证事实一致性，必要时重试`,
    excerpt: "从标准 RAG 到 Agentic RAG，深入分析检索增强生成的架构演进与实践方案。",
    tags: ["AI", "RAG", "Agent", "LLM", "LangChain"], published: true
  },
  {
    id: "seed_post_2", title: "Python 异步编程：从回调到 async/await",
    slug: "python-async-from-callback-to-async-await",
    content: `## 为什么需要异步\n\nPython 的 GIL 限制了多线程表现，对于 IO 密集型任务，异步可极大提升并发性能。\n\n## 核心概念\n\n\`\`\`python\nimport asyncio\n\nasync def fetch_data(url: str):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            return await response.json()\n\`\`\``,
    excerpt: "深入理解 Python 异步编程的演进，从回调到 async/await 的核心概念与最佳实践。",
    tags: ["Python", "异步", "FastAPI", "教程"], published: true
  },
  {
    id: "seed_post_3", title: "分布式系统设计模式：从理论到实践",
    slug: "distributed-system-design-patterns",
    content: `## 核心挑战\n\n- 网络不可靠：消息可能丢失、重复、乱序\n- 时钟不同步：无法依赖全局时钟\n\n## 关键模式\n\n1. 一致性哈希 — 解决数据分片和动态扩缩容\n2. 领导者选举 — Raft/Paxos 保证单领导者\n3. 断路器 — 防止级联故障\n4. CQRS — 读写分离`,
    excerpt: "探索分布式系统核心设计模式：一致性哈希、领导选举、CQRS 等模式的原理与实践。",
    tags: ["架构", "系统设计", "分布式", "数据库"], published: true
  },
  {
    id: "seed_post_4", title: "Transformer 架构深度解析",
    slug: "transformer-architecture-deep-dive",
    content: `## 注意力机制的本质\n\nAttention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V\n\n### 多头注意力\n\n将输入投影到多个子空间，并行计算注意力。\n\n### 位置编码\n\n- 正弦位置编码：原始论文使用\n- 可学习位置编码：BERT 风格\n- 旋转位置编码（RoPE）：LLaMA 等现代模型采用`,
    excerpt: "逐层拆解 Transformer 架构：注意力机制、位置编码、残差连接与前馈网络。",
    tags: ["AI", "Transformer", "NLP", "深度学习"], published: true
  },
  {
    id: "seed_post_5", title: "构建高效的 AI Agent 系统",
    slug: "building-efficient-ai-agent-systems",
    content: `## Agent 架构蓝图\n\n1. 推理引擎 — LLM 驱动的决策核心\n2. 工具集 — 可调用的外部 API/函数\n3. 记忆系统 — 短期和长期记忆\n4. 规划器 — 任务分解与执行排序\n\n## 关键设计原则\n\n- 单一职责：每个 Agent 只负责一个领域\n- 明确接口：Agent 间通过结构化消息通信\n- 优雅降级：工具调用失败时要有备选方案`,
    excerpt: "从架构蓝图到实现细节，构建高效稳定的 AI Agent 系统的完整指南。",
    tags: ["AI", "Agent", "LangChain", "系统设计"], published: true
  },
  {
    id: "seed_post_6", title: "数字花园：培养持久的知识体系",
    slug: "digital-garden-cultivating-knowledge",
    content: `## 什么是数字花园\n\n数字花园是一种不同于传统博客的知识管理方式：\n\n- 博客：按时序排列，追求完美，发布即完成\n- 花园：按主题组织，接受不完美，持续生长\n\n### 幼苗（Seedling）\n刚种下的想法，粗糙但真实。\n\n### 成长（Bud）\n经过思考和补充，开始形成体系。\n\n### 常青（Evergreen）\n成熟的思想，经得起时间考验。`,
    excerpt: "了解数字花园的理念：通过持续培育知识体系实现深度思考与长期成长。",
    tags: ["学习", "成长", "方法论", "PKM"], published: true
  },
].map(p => ({ ...p, createdAt: now(), updatedAt: now() }));

// ─── Seed Projects ─────────────────────────────────────

export const SEED_PROJECTS: Array<Omit<Project, "createdAt" | "updatedAt"> & { createdAt: string; updatedAt: string }> = [
  {
    id: "seed_proj_1", title: "Meta-Learning rPPG 生理感知系统",
    category: "AI / 计算机视觉",
    description: "三支路深度学习架构：自监督预训练 + MAML 元学习 + 证据深度学习，从面部视频非接触式估计心率与呼吸率。",
    tags: ["Python", "PyTorch", "MediaPipe", "MAML", "计算机视觉"],
    featured: false, sortOrder: 0, status: "building",
    githubUrl: "https://github.com/LIU3254621731/meta-learning-rppg", demoUrl: "",
  },
  {
    id: "seed_proj_2", title: "VtuberHub — 全栈虚拟主播套件",
    category: "桌面应用 / 3D",
    description: "跨 WPF、Godot、Unity 的完整虚拟主播工具集，基于 MediaPipe 实现实时 3D 角色驱动与面部捕捉。",
    tags: ["C#", "Godot", "Unity", "WPF", "MediaPipe", "3D"],
    featured: false, sortOrder: 0, status: "building",
    githubUrl: "https://github.com/LIU3254621731/VtuberHub", demoUrl: "",
  },
  {
    id: "seed_proj_3", title: "AI 知识 Wiki 系统",
    category: "全栈 / AI",
    description: "基于 Tauri + Rust 构建的 AI 原生知识管理系统，集成 LLM 检索增强生成与语义搜索。",
    tags: ["Rust", "Tauri", "TypeScript", "LLM", "RAG"],
    featured: false, sortOrder: 0, status: "building",
    githubUrl: "https://github.com/LIU3254621731/llmwiki_dpk", demoUrl: "",
  },
].map(p => ({ ...p, createdAt: now(), updatedAt: now() }));

// ─── Seed Garden Entries ───────────────────────────────

export const SEED_GARDEN: Array<Omit<GardenEntry, "createdAt" | "updatedAt"> & { createdAt: string; updatedAt: string }> = [
  {
    id: "seed_garden_1", title: "为什么我选择公开学习", slug: "why-learn-in-public",
    content: "公开学习是一种强大的成长策略。\n\n## 好处\n\n- 责任驱动：公开承诺推动持续行动\n- 反馈循环：社区反馈加速迭代\n- 建立信任：透明的过程比完美的结果更有说服力",
    excerpt: "公开学习是加速个人成长的最佳策略之一。",
    tags: ["学习", "成长", "方法论"], category: "thought", stage: "evergreen", published: true,
  },
  {
    id: "seed_garden_2", title: "AI 时代的个人知识管理", slug: "pkm-ai-era",
    content: "在 LLM 时代，个人知识管理正在发生根本性变化。\n\n## AI 增强的 PKM\n\n- 自动摘要：AI 提炼长篇内容\n- 语义搜索：超越关键词匹配\n- 知识图谱：自动发现概念间的联系",
    excerpt: "LLM 正在重塑我们管理知识的方式。",
    tags: ["AI", "PKM", "工具"], category: "observation", stage: "bud", published: true,
  },
  {
    id: "seed_garden_3", title: "从学生到 Builder：身份的转变", slug: "student-to-builder",
    content: "大学教育教会我们如何学习，但真正的成长发生在「建造」中。\n\n## Builder 心态\n\n- 主动探索\n- 快速迭代\n- 拥抱失败",
    excerpt: "真正的学习发生在建造的过程中。",
    tags: ["成长", "心态", "创业"], category: "thought", stage: "evergreen", published: true,
  },
].map(g => ({ ...g, createdAt: now(), updatedAt: now() }));

// ─── Seed Site Config ──────────────────────────────────

export const SEED_CONFIG: Record<string, string> = {
  site_title: "Personal Hub",
  site_description: "AI Engineer - Builder - Lifelong Learner",
  about_me: "AI 工程师，热爱构建实用的技术产品。",
  avatar_url: "",
  social_github: "https://github.com/LIU3254621731",
  social_email: "",
};

// ─── Populate in-memory stores ────────────────────────

export function seedMemStores(
  memPosts: any[], memProjects: any[], memGarden: any[],
  memSiteConfig: Record<string, string>,
  memDailyStatus: any, memActivities: any[],
): void {
  if (memPosts.length === 0) memPosts.push(...SEED_POSTS);
  if (memProjects.length === 0) memProjects.push(...SEED_PROJECTS);
  if (memGarden.length === 0) memGarden.push(...SEED_GARDEN);
  if (Object.keys(memSiteConfig).length === 0) Object.assign(memSiteConfig, SEED_CONFIG);
}
