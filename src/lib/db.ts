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

function initTables() {
  const d = getDb();
  d.exec("CREATE TABLE IF NOT EXISTS posts (id TEXT PRIMARY KEY, title TEXT NOT NULL, slug TEXT UNIQUE NOT NULL, content TEXT NOT NULL DEFAULT '', excerpt TEXT DEFAULT '', tags TEXT DEFAULT '[]', published INTEGER DEFAULT 1, created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now')))");
  d.exec("CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, title TEXT NOT NULL, category TEXT NOT NULL DEFAULT '', description TEXT NOT NULL DEFAULT '', tags TEXT DEFAULT '[]', featured INTEGER DEFAULT 0, sort_order INTEGER DEFAULT 0, created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now')))");
  d.exec("CREATE TABLE IF NOT EXISTS site_config (key TEXT PRIMARY KEY, value TEXT NOT NULL)");
}

interface PostRow { id: string; title: string; slug: string; content: string; excerpt: string; tags: string; published: number; created_at: string; updated_at: string; }
interface ProjectRow { id: string; title: string; category: string; description: string; tags: string; featured: number; sort_order: number; created_at: string; updated_at: string; }
export interface Post { id: string; title: string; slug: string; content: string; excerpt: string; tags: string[]; published: boolean; createdAt: string; updatedAt: string; }
export interface Project { id: string; title: string; category: string; description: string; tags: string[]; featured: boolean; sortOrder: number; createdAt: string; updatedAt: string; }

function pPost(r: PostRow): Post {
  return { id: r.id, title: r.title, slug: r.slug, content: r.content, excerpt: r.excerpt, tags: JSON.parse(r.tags || "[]"), published: r.published === 1, createdAt: r.created_at, updatedAt: r.updated_at };
}
function pProj(r: ProjectRow): Project {
  return { id: r.id, title: r.title, category: r.category, description: r.description, tags: JSON.parse(r.tags || "[]"), featured: r.featured === 1, sortOrder: r.sort_order, createdAt: r.created_at, updatedAt: r.updated_at };
}
function genId(): string { return Math.random().toString(36).substring(2, 10) + Date.now().toString(36); }

/* Posts */
export function getPosts(): Post[] {
  return (getDb().prepare("SELECT * FROM posts ORDER BY created_at DESC").all() as PostRow[]).map(pPost);
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
  getDb().prepare("INSERT INTO posts (id,title,slug,content,excerpt,tags,published,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)").run(id, data.title, data.slug, data.content, data.excerpt, JSON.stringify(data.tags), data.published ? 1 : 0, now, now);
  return getPostById(id)!;
}
export function updatePost(id: string, data: Partial<Omit<Post, "id" | "createdAt">>): Post | null {
  const e = getPostById(id); if (!e) return null;
  const now = new Date().toISOString();
  getDb().prepare("UPDATE posts SET title=?,slug=?,content=?,excerpt=?,tags=?,published=?,updated_at=? WHERE id=?").run(data.title ?? e.title, data.slug ?? e.slug, data.content ?? e.content, data.excerpt ?? e.excerpt, JSON.stringify(data.tags ?? e.tags), data.published !== undefined ? (data.published ? 1 : 0) : (e.published ? 1 : 0), now, id);
  return getPostById(id);
}
export function deletePost(id: string): boolean {
  return getDb().prepare("DELETE FROM posts WHERE id = ?").run(id).changes > 0;
}

/* Projects */
export function getProjects(): Project[] {
  return (getDb().prepare("SELECT * FROM projects ORDER BY sort_order ASC, created_at DESC").all() as ProjectRow[]).map(pProj);
}
export function getProjectById(id: string): Project | null {
  const r = getDb().prepare("SELECT * FROM projects WHERE id = ?").get(id) as ProjectRow | undefined;
  return r ? pProj(r) : null;
}
export function createProject(data: Omit<Project, "id" | "createdAt" | "updatedAt">): Project {
  const id = genId(); const now = new Date().toISOString();
  getDb().prepare("INSERT INTO projects (id,title,category,description,tags,featured,sort_order,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)").run(id, data.title, data.category, data.description, JSON.stringify(data.tags), data.featured ? 1 : 0, data.sortOrder ?? 0, now, now);
  return getProjectById(id)!;
}
export function updateProject(id: string, data: Partial<Omit<Project, "id" | "createdAt">>): Project | null {
  const e = getProjectById(id); if (!e) return null;
  const now = new Date().toISOString();
  getDb().prepare("UPDATE projects SET title=?,category=?,description=?,tags=?,featured=?,sort_order=?,updated_at=? WHERE id=?").run(data.title ?? e.title, data.category ?? e.category, data.description ?? e.description, JSON.stringify(data.tags ?? e.tags), data.featured !== undefined ? (data.featured ? 1 : 0) : (e.featured ? 1 : 0), data.sortOrder ?? e.sortOrder, now, id);
  return getProjectById(id);
}
export function deleteProject(id: string): boolean {
  return getDb().prepare("DELETE FROM projects WHERE id = ?").run(id).changes > 0;
}

/* Site Config */
export function getSiteConfig(): Record<string, string> {
  const rows = getDb().prepare("SELECT * FROM site_config").all() as { key: string; value: string }[];
  const c: Record<string, string> = {};
  for (const r of rows) c[r.key] = r.value;
  return c;
}
export function setSiteConfig(key: string, value: string): void {
  getDb().prepare("INSERT OR REPLACE INTO site_config (key, value) VALUES (?, ?)").run(key, value);
}
export function setSiteConfigs(data: Record<string, string>): void {
  const s = getDb().prepare("INSERT OR REPLACE INTO site_config (key, value) VALUES (?, ?)");
  getDb().transaction(() => { for (const [k, v] of Object.entries(data)) s.run(k, v); })();
}

/* Seed */
function seedIfEmpty() {
  const c = (getDb().prepare("SELECT COUNT(*) as count FROM posts").get() as { count: number }).count;
  if (c > 0) return;
  const now = new Date().toISOString();
  createPost({ title: "从研究到产品：构建非接触式心率监测器", slug: "rppg-journey", content: "从学术研究到精致的桌面产品，这条路从来不平坦。\n\n## 研究基础\n\n我的工作始于复现 Meta-rPPG（ECCV 2020），使用转导式元学习（MAML）从面部视频估计心率。\n\n## 从 Notebook 到应用\n\n桌面端采用生产者-消费者架构，PyQt5 渲染实时波形。\n\n## 关键收获\n\n- 实时信号处理需精心管理延迟\n- Python GIL 下的多线程架构\n- 学术代码需大量重构才能用于生产", excerpt: "如何将学术 rPPG 研究转化为生产级桌面应用。", tags: ["rPPG", "计算机视觉", "产品"], published: true });
  createPost({ title: "元学习生理感知：融合 SimCLR、MAML 与证据深度学习", slug: "meta-learning-rppg", content: "远程生理感知是极具挑战的信号提取问题。\n\n## 三分支架构\n\n### 自监督预训练\nSimCLR 风格时间对比目标预训练 Transformer。\n\n### 元学习自适应\nMAML 跨受试者快速适应。\n\n### 证据回归\nNIG 分布输出校准不确定性。", excerpt: "深入解析三分支架构。", tags: ["深度学习", "rPPG", "研究"], published: true });
  createProject({ title: "rPPG 远程生理感知系统", category: "AI / 计算机视觉", description: "通过普通摄像头非接触式测量心率和呼吸率。", tags: ["Python", "PyTorch", "MediaPipe"], featured: true, sortOrder: 0 });
  createProject({ title: "VtuberHub — 全栈虚拟主播套件", category: "桌面应用 / 3D", description: "跨 WPF、Godot、Unity 的完整虚拟主播工具集。", tags: ["C#", "Godot", "Unity"], featured: true, sortOrder: 1 });
  createProject({ title: "LLM 智能知识 Wiki", category: "全栈 / AI", description: "基于 Tauri + Rust 构建的 AI 原生知识管理系统。", tags: ["Rust", "Tauri", "TypeScript"], featured: true, sortOrder: 2 });
}
