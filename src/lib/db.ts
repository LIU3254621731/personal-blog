import Database from "better-sqlite3";
import path from "path";
import fs from "fs";

const DB_PATH = path.join(process.cwd(), "data", "blog.db");

let db: Database.Database;
let seeded = false;

function getDb(): Database.Database {
  if (!db) {
    const dir = path.dirname(DB_PATH);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    db = new Database(DB_PATH);
    db.pragma("journal_mode = WAL");
    db.pragma("busy_timeout = 5000");
    initTables();
    if (!seeded) {
      seeded = true;
      seedIfEmpty();
    }
  }
  return db;
}

function columnExists(table: string, column: string): boolean {
  const rows = getDb()
    .prepare(`PRAGMA table_info(${table})`)
    .all() as { name: string }[];
  return rows.some((r) => r.name === column);
}

function initTables() {
  const d = getDb();

  d.exec(`CREATE TABLE IF NOT EXISTS posts (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    excerpt TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',
    published INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
  )`);

  d.exec(`CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    tags TEXT DEFAULT '[]',
    featured INTEGER DEFAULT 0,
    sort_order INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
  )`);

  // Migrate: add new columns to projects (safe to run on existing DBs)
  if (!columnExists("projects", "status")) {
    d.exec(`ALTER TABLE projects ADD COLUMN status TEXT DEFAULT 'building'`);
  }
  if (!columnExists("projects", "github_url")) {
    d.exec(`ALTER TABLE projects ADD COLUMN github_url TEXT DEFAULT ''`);
  }
  if (!columnExists("projects", "demo_url")) {
    d.exec(`ALTER TABLE projects ADD COLUMN demo_url TEXT DEFAULT ''`);
  }

  d.exec(`CREATE TABLE IF NOT EXISTS site_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
  )`);

  d.exec(`CREATE TABLE IF NOT EXISTS daily_status (
    id TEXT PRIMARY KEY,
    learning TEXT DEFAULT '',
    building TEXT DEFAULT '',
    reading TEXT DEFAULT '',
    thinking TEXT DEFAULT '',
    updated_at TEXT DEFAULT (datetime('now'))
  )`);

  d.exec(`CREATE TABLE IF NOT EXISTS learning_activity (
    id TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    count INTEGER DEFAULT 1,
    UNIQUE(date)
  )`);

  d.exec(`CREATE TABLE IF NOT EXISTS garden_entries (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    excerpt TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',
    category TEXT DEFAULT 'thought',
    stage TEXT DEFAULT 'seedling',
    published INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
  )`);
}

/* ---- Row types ---- */
interface PostRow {
  id: string; title: string; slug: string; content: string; excerpt: string;
  tags: string; published: number; created_at: string; updated_at: string;
}
interface ProjectRow {
  id: string; title: string; category: string; description: string;
  tags: string; featured: number; sort_order: number;
  status?: string; github_url?: string; demo_url?: string;
  created_at: string; updated_at: string;
}
interface GardenRow {
  id: string; title: string; slug: string; content: string; excerpt: string;
  tags: string; category: string; stage: string; published: number;
  created_at: string; updated_at: string;
}

/* ---- Public interfaces ---- */
export interface Post {
  id: string; title: string; slug: string; content: string; excerpt: string;
  tags: string[]; published: boolean; createdAt: string; updatedAt: string;
}

export interface Project {
  id: string; title: string; category: string; description: string;
  tags: string[]; featured: boolean; sortOrder: number;
  status: string; githubUrl: string; demoUrl: string;
  createdAt: string; updatedAt: string;
}

export interface DailyStatus {
  id: string;
  learning: string;
  building: string;
  reading: string;
  thinking: string;
  updatedAt: string;
}

export interface LearningActivity {
  id: string;
  date: string;
  count: number;
}

export interface GardenEntry {
  id: string; title: string; slug: string; content: string; excerpt: string;
  tags: string[]; category: string; stage: string; published: boolean;
  createdAt: string; updatedAt: string;
}

/* ---- Parsers ---- */
function pPost(r: PostRow): Post {
  return {
    id: r.id, title: r.title, slug: r.slug, content: r.content,
    excerpt: r.excerpt, tags: JSON.parse(r.tags || "[]"),
    published: r.published === 1, createdAt: r.created_at, updatedAt: r.updated_at,
  };
}
function pProj(r: ProjectRow): Project {
  return {
    id: r.id, title: r.title, category: r.category, description: r.description,
    tags: JSON.parse(r.tags || "[]"),
    featured: r.featured === 1, sortOrder: r.sort_order ?? 0,
    status: r.status ?? "building",
    githubUrl: r.github_url ?? "",
    demoUrl: r.demo_url ?? "",
    createdAt: r.created_at, updatedAt: r.updated_at,
  };
}
function pGarden(r: GardenRow): GardenEntry {
  return {
    id: r.id, title: r.title, slug: r.slug, content: r.content,
    excerpt: r.excerpt, tags: JSON.parse(r.tags || "[]"),
    category: r.category, stage: r.stage,
    published: r.published === 1,
    createdAt: r.created_at, updatedAt: r.updated_at,
  };
}

function genId(): string {
  return Math.random().toString(36).substring(2, 10) + Date.now().toString(36);
}

/* ========== Posts ========== */
export function getPosts(): Post[] {
  return (getDb()
    .prepare("SELECT * FROM posts ORDER BY created_at DESC")
    .all() as PostRow[]).map(pPost);
}
export function getPostBySlug(slug: string): Post | null {
  const r = getDb().prepare("SELECT * FROM posts WHERE slug = ?").get(slug) as PostRow | undefined;
  return r ? pPost(r) : null;
}
export function getPostById(id: string): Post | null {
  const r = getDb().prepare("SELECT * FROM posts WHERE id = ?").get(id) as PostRow | undefined;
  return r ? pPost(r) : null;
}
export function createPost(data: Omit<Post, "id" | "createdAt" | "updatedAt">): Post {
  const id = genId(); const now = new Date().toISOString();
  getDb()
    .prepare("INSERT INTO posts (id,title,slug,content,excerpt,tags,published,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)")
    .run(id, data.title, data.slug, data.content, data.excerpt, JSON.stringify(data.tags), data.published ? 1 : 0, now, now);
  return getPostById(id)!;
}
export function updatePost(id: string, data: Partial<Omit<Post, "id" | "createdAt">>): Post | null {
  const e = getPostById(id); if (!e) return null;
  const now = new Date().toISOString();
  getDb()
    .prepare("UPDATE posts SET title=?,slug=?,content=?,excerpt=?,tags=?,published=?,updated_at=? WHERE id=?")
    .run(data.title ?? e.title, data.slug ?? e.slug, data.content ?? e.content, data.excerpt ?? e.excerpt, JSON.stringify(data.tags ?? e.tags), data.published !== undefined ? (data.published ? 1 : 0) : (e.published ? 1 : 0), now, id);
  return getPostById(id);
}
export function deletePost(id: string): boolean {
  return getDb().prepare("DELETE FROM posts WHERE id = ?").run(id).changes > 0;
}

/* ========== Projects ========== */
export function getProjects(): Project[] {
  return (getDb()
    .prepare("SELECT * FROM projects ORDER BY sort_order ASC, created_at DESC")
    .all() as ProjectRow[]).map(pProj);
}
export function getProjectById(id: string): Project | null {
  const r = getDb().prepare("SELECT * FROM projects WHERE id = ?").get(id) as ProjectRow | undefined;
  return r ? pProj(r) : null;
}
export function createProject(data: Omit<Project, "id" | "createdAt" | "updatedAt">): Project {
  const id = genId(); const now = new Date().toISOString();
  getDb()
    .prepare("INSERT INTO projects (id,title,category,description,tags,featured,sort_order,status,github_url,demo_url,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)")
    .run(id, data.title, data.category, data.description, JSON.stringify(data.tags), data.featured ? 1 : 0, data.sortOrder ?? 0, data.status ?? "building", data.githubUrl ?? "", data.demoUrl ?? "", now, now);
  return getProjectById(id)!;
}
export function updateProject(id: string, data: Partial<Omit<Project, "id" | "createdAt">>): Project | null {
  const e = getProjectById(id); if (!e) return null;
  const now = new Date().toISOString();
  getDb()
    .prepare("UPDATE projects SET title=?,category=?,description=?,tags=?,featured=?,sort_order=?,status=?,github_url=?,demo_url=?,updated_at=? WHERE id=?")
    .run(data.title ?? e.title, data.category ?? e.category, data.description ?? e.description, JSON.stringify(data.tags ?? e.tags), data.featured !== undefined ? (data.featured ? 1 : 0) : (e.featured ? 1 : 0), data.sortOrder ?? e.sortOrder, data.status ?? e.status, data.githubUrl ?? e.githubUrl, data.demoUrl ?? e.demoUrl, now, id);
  return getProjectById(id);
}
export function deleteProject(id: string): boolean {
  return getDb().prepare("DELETE FROM projects WHERE id = ?").run(id).changes > 0;
}

/* ========== Site Config ========== */
export function getSiteConfig(): Record<string, string> {
  const rows = getDb()
    .prepare("SELECT * FROM site_config")
    .all() as { key: string; value: string }[];
  const c: Record<string, string> = {};
  for (const r of rows) c[r.key] = r.value;
  return c;
}
export function setSiteConfig(key: string, value: string): void {
  getDb()
    .prepare("INSERT OR REPLACE INTO site_config (key, value) VALUES (?, ?)")
    .run(key, value);
}
export function setSiteConfigs(data: Record<string, string>): void {
  const s = getDb().prepare("INSERT OR REPLACE INTO site_config (key, value) VALUES (?, ?)");
  getDb().transaction(() => { for (const [k, v] of Object.entries(data)) s.run(k, v); })();
}
export function deleteSiteConfig(key: string): boolean {
  return getDb().prepare("DELETE FROM site_config WHERE key = ?").run(key).changes > 0;
}

/* ========== Daily Status ========== */
export function getDailyStatus(): DailyStatus | null {
  const r = getDb()
    .prepare("SELECT * FROM daily_status ORDER BY updated_at DESC LIMIT 1")
    .get() as { id: string; learning: string; building: string; reading: string; thinking: string; updated_at: string } | undefined;
  if (!r) return null;
  return { id: r.id, learning: r.learning, building: r.building, reading: r.reading, thinking: r.thinking, updatedAt: r.updated_at };
}
export function updateDailyStatus(data: Omit<DailyStatus, "id" | "updatedAt">): DailyStatus {
  const d = getDb();
  const existing = d.prepare("SELECT id FROM daily_status ORDER BY updated_at DESC LIMIT 1").get() as { id: string } | undefined;
  const now = new Date().toISOString();
  if (existing) {
    d.prepare("UPDATE daily_status SET learning=?,building=?,reading=?,thinking=?,updated_at=? WHERE id=?")
      .run(data.learning, data.building, data.reading, data.thinking, now, existing.id);
    return getDailyStatus()!;
  }
  const id = genId();
  d.prepare("INSERT INTO daily_status (id,learning,building,reading,thinking,updated_at) VALUES (?,?,?,?,?,?)")
    .run(id, data.learning, data.building, data.reading, data.thinking, now);
  return getDailyStatus()!;
}

/* ========== Learning Activity ========== */
export function getLearningActivities(): LearningActivity[] {
  return (getDb()
    .prepare("SELECT * FROM learning_activity ORDER BY date ASC")
    .all() as { id: string; date: string; count: number }[]);
}
export function logLearningActivity(date: string, count: number = 1): void {
  getDb()
    .prepare("INSERT INTO learning_activity (id, date, count) VALUES (?, ?, ?) ON CONFLICT(date) DO UPDATE SET count = count + ?")
    .run(genId(), date, count, count);
}

/* ========== Garden Entries ========== */
export function getGardenEntries(): GardenEntry[] {
  return (getDb()
    .prepare("SELECT * FROM garden_entries WHERE published = 1 ORDER BY created_at DESC")
    .all() as GardenRow[]).map(pGarden);
}
export function getGardenEntryBySlug(slug: string): GardenEntry | null {
  const r = getDb().prepare("SELECT * FROM garden_entries WHERE slug = ? AND published = 1").get(slug) as GardenRow | undefined;
  return r ? pGarden(r) : null;
}
export function getGardenEntryById(id: string): GardenEntry | null {
  const r = getDb().prepare("SELECT * FROM garden_entries WHERE id = ?").get(id) as GardenRow | undefined;
  return r ? pGarden(r) : null;
}
export function createGardenEntry(data: Omit<GardenEntry, "id" | "createdAt" | "updatedAt">): GardenEntry {
  const id = genId(); const now = new Date().toISOString();
  getDb()
    .prepare("INSERT INTO garden_entries (id,title,slug,content,excerpt,tags,category,stage,published,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)")
    .run(id, data.title, data.slug, data.content, data.excerpt, JSON.stringify(data.tags), data.category, data.stage, data.published ? 1 : 0, now, now);
  return getGardenEntryById(id)!;
}
export function updateGardenEntry(id: string, data: Partial<Omit<GardenEntry, "id" | "createdAt">>): GardenEntry | null {
  const e = getGardenEntryById(id); if (!e) return null;
  const now = new Date().toISOString();
  getDb()
    .prepare("UPDATE garden_entries SET title=?,slug=?,content=?,excerpt=?,tags=?,category=?,stage=?,published=?,updated_at=? WHERE id=?")
    .run(data.title ?? e.title, data.slug ?? e.slug, data.content ?? e.content, data.excerpt ?? e.excerpt, JSON.stringify(data.tags ?? e.tags), data.category ?? e.category, data.stage ?? e.stage, data.published !== undefined ? (data.published ? 1 : 0) : (e.published ? 1 : 0), now, id);
  return getGardenEntryById(id);
}
export function deleteGardenEntry(id: string): boolean {
  return getDb().prepare("DELETE FROM garden_entries WHERE id = ?").run(id).changes > 0;
}

/* ========== Seed ========== */
function seedIfEmpty() {
  const d = getDb();

  // Seed posts if empty
  const postCount = (d.prepare("SELECT COUNT(*) as count FROM posts").get() as { count: number }).count;
  if (postCount === 0) {
    createPost({
      title: "从研究到产品：构建非接触式心率监测器", slug: "rppg-journey",
      content:
        "从学术研究到精致的桌面产品，这条路从来不平坦。\n\n## 研究基础\n\n我的工作始于复现 Meta-rPPG（ECCV 2020），使用转导式元学习（MAML）从面部视频估计心率。\n\n## 从 Notebook 到应用\n\n桌面端采用生产者-消费者架构，PyQt5 渲染实时波形。\n\n## 关键收获\n\n- 实时信号处理需精心管理延迟\n- Python GIL 下的多线程架构\n- 学术代码需大量重构才能用于生产",
      excerpt: "如何将学术 rPPG 研究转化为生产级桌面应用。",
      tags: ["rPPG", "计算机视觉", "产品"], published: true,
    });
    createPost({
      title: "元学习生理感知：融合 SimCLR、MAML 与证据深度学习", slug: "meta-learning-rppg",
      content:
        "远程生理感知是极具挑战的信号提取问题。\n\n## 三分支架构\n\n### 自监督预训练\nSimCLR 风格时间对比目标预训练 Transformer。\n\n### 元学习自适应\nMAML 跨受试者快速适应。\n\n### 证据回归\nNIG 分布输出校准不确定性。",
      excerpt: "深入解析三分支架构。",
      tags: ["深度学习", "rPPG", "研究"], published: true,
    });
  }

  // Seed projects if empty
  const projectCount = (d.prepare("SELECT COUNT(*) as count FROM projects").get() as { count: number }).count;
  if (projectCount === 0) {
    createProject({
      title: "rPPG 远程生理感知系统", category: "AI",
      description: "通过普通摄像头非接触式测量心率和呼吸率。",
      tags: ["Python", "PyTorch", "MediaPipe"], featured: true, sortOrder: 0,
      status: "released", githubUrl: "https://github.com", demoUrl: "",
    });
    createProject({
      title: "VtuberHub — 全栈虚拟主播套件", category: "Web",
      description: "跨 WPF、Godot、Unity 的完整虚拟主播工具集。",
      tags: ["C#", "Godot", "Unity"], featured: true, sortOrder: 1,
      status: "building", githubUrl: "https://github.com", demoUrl: "",
    });
    createProject({
      title: "LLM 智能知识 Wiki", category: "AI",
      description: "基于 Tauri + Rust 构建的 AI 原生知识管理系统。",
      tags: ["Rust", "Tauri", "TypeScript"], featured: true, sortOrder: 2,
      status: "building", githubUrl: "https://github.com", demoUrl: "",
    });
  }

  // Seed daily status if empty
  const statusCount = (d.prepare("SELECT COUNT(*) as count FROM daily_status").get() as { count: number }).count;
  if (statusCount === 0) {
    updateDailyStatus({
      learning: "Transformer Architecture",
      building: "LLM Wiki — 知识管理系统",
      reading: "《深度学习入门》",
      thinking: "AI Agent 的自主决策边界",
    });
  }

  // Seed learning activity if empty
  const activityCount = (d.prepare("SELECT COUNT(*) as count FROM learning_activity").get() as { count: number }).count;
  if (activityCount === 0) {
    const now = new Date();
    for (let i = 365; i >= 0; i--) {
      const d = new Date(now);
      d.setDate(d.getDate() - i);
      const dateStr = d.toISOString().split("T")[0];
      const dayOfWeek = d.getDay();
      const baseActivity = dayOfWeek >= 1 && dayOfWeek <= 5 ? 0.6 : 0.3;
      const count = Math.random() < baseActivity ? Math.floor(Math.random() * 4) + 1 : 0;
      if (count > 0) {
        logLearningActivity(dateStr, count);
      }
    }
  }

  // Seed garden entries if empty
  const gardenCount = (d.prepare("SELECT COUNT(*) as count FROM garden_entries").get() as { count: number }).count;
  if (gardenCount === 0) {
    createGardenEntry({
      title: "为什么我选择公开学习",
      slug: "why-learn-in-public",
      content:
        "公开学习是一种强大的成长策略。\n\n## 好处\n\n- **责任驱动**：公开承诺推动持续行动\n- **反馈循环**：社区反馈加速迭代\n- **建立信任**：透明的过程比完美的结果更有说服力\n\n## 开始的方式\n\n不需要等「准备好」，现在就可以开始。记录你的思考，分享你的失败，展示你的进展。",
      excerpt: "公开学习是加速个人成长的最佳策略之一。",
      tags: ["学习", "成长", "方法论"], category: "thought", stage: "evergreen", published: true,
    });
    createGardenEntry({
      title: "AI 时代的个人知识管理",
      slug: "pkm-ai-era",
      content:
        "在 LLM 时代，个人知识管理正在发生根本性变化。\n\n## 传统 PKM 的局限\n\n文件夹、标签、双向链接——这些方法仍然有效，但已经不够。\n\n## AI 增强的 PKM\n\n- **自动摘要**：AI 提炼长篇内容\n- **语义搜索**：超越关键词匹配\n- **知识图谱**：自动发现概念间的联系\n- **智能推荐**：推送相关的历史笔记",
      excerpt: "LLM 正在重塑我们管理知识的方式。",
      tags: ["AI", "PKM", "工具"], category: "observation", stage: "bud", published: true,
    });
    createGardenEntry({
      title: "从学生到 Builder：身份的转变",
      slug: "student-to-builder",
      content:
        "大学教育教会我们如何学习，但真正的成长发生在「建造」中。\n\n## 学生心态\n\n- 等待指导\n- 追求完美\n- 害怕犯错\n\n## Builder 心态\n\n- 主动探索\n- 快速迭代\n- 拥抱失败\n\n## 转变的关键\n\n开始做项目。不是为了分数，而是为了创造一个能用的东西。",
      excerpt: "真正的学习发生在建造的过程中。",
      tags: ["成长", "心态", "创业"], category: "thought", stage: "evergreen", published: true,
    });
  }
}
