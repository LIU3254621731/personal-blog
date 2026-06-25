/**
 * Database module — SQLite (local) with graceful fallback (EdgeOne).
 *
 * When better-sqlite3 is available (local dev), full CRUD with persistence.
 * On EdgeOne (no native modules), falls back to in-memory arrays — pages
 * render with empty/default data, admin operations for Roadmap/Resources
 * use site_config. Full CloudBase integration comes in Phase 2.
 */

import { seedMemStores } from "./seed";

// ─── Try loading SQLite, fall back gracefully ─────────

let db: any = null;
let dbAvailable = false;

try {
  const Database = require("better-sqlite3");
  const path = require("path");
  const fs = require("fs");
  const DB_PATH = path.join(process.cwd(), "data", "blog.db");
  const dir = path.dirname(DB_PATH);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  db = new Database(DB_PATH);
  db.pragma("journal_mode = WAL");
  db.pragma("busy_timeout = 5000");
  dbAvailable = true;
  console.log("[db] SQLite loaded:", DB_PATH);
} catch (e: any) {
  console.warn("[db] SQLite unavailable, using in-memory fallback:", e.message?.substring(0, 80));
}

// ─── In-memory fallback stores ────────────────────────

interface MemPost {
  id: string; title: string; slug: string; content: string; excerpt: string;
  tags: string[]; published: boolean; createdAt: string; updatedAt: string;
}
interface MemProject {
  id: string; title: string; category: string; description: string;
  tags: string[]; featured: boolean; sortOrder: number;
  status: string; githubUrl: string; demoUrl: string;
  createdAt: string; updatedAt: string;
}
interface MemGarden {
  id: string; title: string; slug: string; content: string; excerpt: string;
  tags: string[]; category: string; stage: string; published: boolean;
  createdAt: string; updatedAt: string;
}

let memPosts: MemPost[] = [];
let memProjects: MemProject[] = [];
let memGarden: MemGarden[] = [];
let memSiteConfig: Record<string, string> = {};
let memDailyStatus: any = null;
let memActivities: { id: string; date: string; count: number }[] = [];

// ─── Seed fallback (EdgeOne) ──────────────────────────
let _seeded = false;
function ensureSeed() {
  if (!dbAvailable && !_seeded) {
    _seeded = true;
    seedMemStores(memPosts, memProjects, memGarden, memSiteConfig, memDailyStatus, memActivities);
  }
}

// ─── SQLite helpers (only used when dbAvailable) ──────

function initTables() {
  if (!db || !dbAvailable) return;
  db.exec(`CREATE TABLE IF NOT EXISTS posts (id TEXT PRIMARY KEY, title TEXT NOT NULL, slug TEXT UNIQUE NOT NULL, content TEXT DEFAULT '', excerpt TEXT DEFAULT '', tags TEXT DEFAULT '[]', published INTEGER DEFAULT 1, created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now')))`);
  db.exec(`CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, title TEXT NOT NULL, category TEXT DEFAULT '', description TEXT DEFAULT '', tags TEXT DEFAULT '[]', featured INTEGER DEFAULT 0, sort_order INTEGER DEFAULT 0, status TEXT DEFAULT 'building', github_url TEXT DEFAULT '', demo_url TEXT DEFAULT '', created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now')))`);
  db.exec(`CREATE TABLE IF NOT EXISTS site_config (key TEXT PRIMARY KEY, value TEXT NOT NULL)`);
  db.exec(`CREATE TABLE IF NOT EXISTS daily_status (id TEXT PRIMARY KEY, learning TEXT DEFAULT '', building TEXT DEFAULT '', reading TEXT DEFAULT '', thinking TEXT DEFAULT '', updated_at TEXT DEFAULT (datetime('now')))`);
  db.exec(`CREATE TABLE IF NOT EXISTS learning_activity (id TEXT PRIMARY KEY, date TEXT NOT NULL, count INTEGER DEFAULT 1, UNIQUE(date))`);
  db.exec(`CREATE TABLE IF NOT EXISTS garden_entries (id TEXT PRIMARY KEY, title TEXT NOT NULL, slug TEXT UNIQUE NOT NULL, content TEXT DEFAULT '', excerpt TEXT DEFAULT '', tags TEXT DEFAULT '[]', category TEXT DEFAULT 'thought', stage TEXT DEFAULT 'seedling', published INTEGER DEFAULT 1, created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now')))`);
}

function genId(): string {
  return Math.random().toString(36).substring(2, 10) + Date.now().toString(36);
}

// ─── Public types ─────────────────────────────────────

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
  id: string; learning: string; building: string; reading: string; thinking: string; updatedAt: string;
}
export interface LearningActivity {
  id: string; date: string; count: number;
}
export interface GardenEntry {
  id: string; title: string; slug: string; content: string; excerpt: string;
  tags: string[]; category: string; stage: string; published: boolean;
  createdAt: string; updatedAt: string;
}

// ─── Posts ────────────────────────────────────────────

// Define the type for a post row from SQLite
interface PostRow { id: string; title: string; slug: string; content: string; excerpt: string; tags: string; published: number; created_at: string; updated_at: string; }

export function getPosts(): Post[] {
  if (dbAvailable) {
    return (db.prepare("SELECT * FROM posts ORDER BY created_at DESC").all() as PostRow[]).map(r => ({
      id: r.id, title: r.title, slug: r.slug, content: r.content,
      excerpt: r.excerpt, tags: JSON.parse(r.tags || "[]"),
      published: r.published === 1, createdAt: r.created_at, updatedAt: r.updated_at,
    }));
  }
  return memPosts;
}

export function getPostBySlug(slug: string): Post | null {
  if (dbAvailable) {
    const r = db.prepare("SELECT * FROM posts WHERE slug = ?").get(slug) as PostRow | undefined;
    return r ? { id: r.id, title: r.title, slug: r.slug, content: r.content, excerpt: r.excerpt, tags: JSON.parse(r.tags || "[]"), published: r.published === 1, createdAt: r.created_at, updatedAt: r.updated_at } : null;
  }
  return memPosts.find(p => p.slug === slug) || null;
}

export function getPostById(id: string): Post | null {
  if (dbAvailable) {
    const r = db.prepare("SELECT * FROM posts WHERE id = ?").get(id) as PostRow | undefined;
    return r ? { id: r.id, title: r.title, slug: r.slug, content: r.content, excerpt: r.excerpt, tags: JSON.parse(r.tags || "[]"), published: r.published === 1, createdAt: r.created_at, updatedAt: r.updated_at } : null;
  }
  return memPosts.find(p => p.id === id) || null;
}

export function createPost(data: Omit<Post, "id" | "createdAt" | "updatedAt">): Post {
  if (dbAvailable) {
    const id = genId(); const now = new Date().toISOString();
    db.prepare("INSERT INTO posts (id,title,slug,content,excerpt,tags,published,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)")
      .run(id, data.title, data.slug, data.content, data.excerpt, JSON.stringify(data.tags), data.published ? 1 : 0, now, now);
    return getPostById(id)!;
  }
  const post: MemPost = { ...data, id: genId(), createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() };
  memPosts.unshift(post);
  return post;
}

export function updatePost(id: string, data: Partial<Omit<Post, "id" | "createdAt">>): Post | null {
  if (dbAvailable) {
    const e = getPostById(id); if (!e) return null;
    const now = new Date().toISOString();
    db.prepare("UPDATE posts SET title=?,slug=?,content=?,excerpt=?,tags=?,published=?,updated_at=? WHERE id=?")
      .run(data.title ?? e.title, data.slug ?? e.slug, data.content ?? e.content, data.excerpt ?? e.excerpt, JSON.stringify(data.tags ?? e.tags), data.published !== undefined ? (data.published ? 1 : 0) : (e.published ? 1 : 0), now, id);
    return getPostById(id);
  }
  const idx = memPosts.findIndex(p => p.id === id);
  if (idx < 0) return null;
  Object.assign(memPosts[idx], data);
  memPosts[idx].updatedAt = new Date().toISOString();
  return memPosts[idx];
}

export function deletePost(id: string): boolean {
  if (dbAvailable) return db.prepare("DELETE FROM posts WHERE id = ?").run(id).changes > 0;
  const idx = memPosts.findIndex(p => p.id === id);
  if (idx >= 0) { memPosts.splice(idx, 1); return true; }
  return false;
}

// ─── Projects ─────────────────────────────────────────

interface ProjectRow { id: string; title: string; category: string; description: string; tags: string; featured: number; sort_order: number; status: string; github_url: string; demo_url: string; created_at: string; updated_at: string; }

export function getProjects(): Project[] {
  if (dbAvailable) {
    return (db.prepare("SELECT * FROM projects ORDER BY sort_order ASC, created_at DESC").all() as ProjectRow[]).map(r => ({
      id: r.id, title: r.title, category: r.category, description: r.description,
      tags: JSON.parse(r.tags || "[]"), featured: r.featured === 1, sortOrder: r.sort_order,
      status: r.status || "building", githubUrl: r.github_url || "", demoUrl: r.demo_url || "",
      createdAt: r.created_at, updatedAt: r.updated_at,
    }));
  }
  return memProjects;
}

export function getProjectById(id: string): Project | null {
  if (dbAvailable) {
    const r = db.prepare("SELECT * FROM projects WHERE id = ?").get(id) as ProjectRow | undefined;
    return r ? {
      id: r.id, title: r.title, category: r.category, description: r.description,
      tags: JSON.parse(r.tags || "[]"), featured: r.featured === 1, sortOrder: r.sort_order,
      status: r.status || "building", githubUrl: r.github_url || "", demoUrl: r.demo_url || "",
      createdAt: r.created_at, updatedAt: r.updated_at,
    } : null;
  }
  return memProjects.find(p => p.id === id) || null;
}

export function createProject(data: Omit<Project, "id" | "createdAt" | "updatedAt">): Project {
  if (dbAvailable) {
    const id = genId(); const now = new Date().toISOString();
    db.prepare("INSERT INTO projects (id,title,category,description,tags,featured,sort_order,status,github_url,demo_url,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)")
      .run(id, data.title, data.category, data.description, JSON.stringify(data.tags), data.featured ? 1 : 0, data.sortOrder, data.status || "building", data.githubUrl || "", data.demoUrl || "", now, now);
    return getProjectById(id)!;
  }
  const p: MemProject = { ...data, id: genId(), createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() };
  memProjects.push(p);
  return p;
}

export function updateProject(id: string, data: Partial<Omit<Project, "id" | "createdAt">>): Project | null {
  if (dbAvailable) {
    const e = getProjectById(id); if (!e) return null;
    const now = new Date().toISOString();
    db.prepare("UPDATE projects SET title=?,category=?,description=?,tags=?,featured=?,sort_order=?,status=?,github_url=?,demo_url=?,updated_at=? WHERE id=?")
      .run(data.title ?? e.title, data.category ?? e.category, data.description ?? e.description, JSON.stringify(data.tags ?? e.tags), data.featured !== undefined ? (data.featured ? 1 : 0) : (e.featured ? 1 : 0), data.sortOrder ?? e.sortOrder, data.status ?? e.status, data.githubUrl ?? e.githubUrl, data.demoUrl ?? e.demoUrl, now, id);
    return getProjectById(id);
  }
  const idx = memProjects.findIndex(p => p.id === id);
  if (idx < 0) return null;
  Object.assign(memProjects[idx], data);
  memProjects[idx].updatedAt = new Date().toISOString();
  return memProjects[idx];
}

export function deleteProject(id: string): boolean {
  if (dbAvailable) return db.prepare("DELETE FROM projects WHERE id = ?").run(id).changes > 0;
  const idx = memProjects.findIndex(p => p.id === id);
  if (idx >= 0) { memProjects.splice(idx, 1); return true; }
  return false;
}

// ─── Site Config ──────────────────────────────────────

export function getSiteConfig(): Record<string, string> {
  if (dbAvailable) {
    const rows = db.prepare("SELECT * FROM site_config").all() as { key: string; value: string }[];
    const c: Record<string, string> = {};
    for (const r of rows) c[r.key] = r.value;
    return c;
  }
  return { ...memSiteConfig };
}

export function setSiteConfig(key: string, value: string): void {
  if (dbAvailable) {
    db.prepare("INSERT OR REPLACE INTO site_config (key, value) VALUES (?, ?)").run(key, value);
    return;
  }
  memSiteConfig[key] = value;
}

export function setSiteConfigs(data: Record<string, string>): void {
  if (dbAvailable) {
    const s = db.prepare("INSERT OR REPLACE INTO site_config (key, value) VALUES (?, ?)");
    const t = db.transaction(() => { for (const [k, v] of Object.entries(data)) s.run(k, v); });
    t();
    return;
  }
  Object.assign(memSiteConfig, data);
}

export function deleteSiteConfig(key: string): void {
  if (dbAvailable) {
    db.prepare("DELETE FROM site_config WHERE key = ?").run(key);
    return;
  }
  delete memSiteConfig[key];
}

// ─── Daily Status ─────────────────────────────────────

export function getDailyStatus(): DailyStatus | null {
  if (dbAvailable) {
    const r = db.prepare("SELECT * FROM daily_status ORDER BY updated_at DESC LIMIT 1").get() as any;
    if (!r) return null;
    return { id: r.id, learning: r.learning, building: r.building, reading: r.reading, thinking: r.thinking, updatedAt: r.updated_at };
  }
  return memDailyStatus;
}

export function updateDailyStatus(data: Omit<DailyStatus, "id" | "updatedAt">): DailyStatus {
  if (dbAvailable) {
    const existing = db.prepare("SELECT id FROM daily_status ORDER BY updated_at DESC LIMIT 1").get() as { id: string } | undefined;
    const now = new Date().toISOString();
    if (existing) {
      db.prepare("UPDATE daily_status SET learning=?,building=?,reading=?,thinking=?,updated_at=? WHERE id=?")
        .run(data.learning, data.building, data.reading, data.thinking, now, existing.id);
      return getDailyStatus()!;
    }
    const id = genId();
    db.prepare("INSERT INTO daily_status (id,learning,building,reading,thinking,updated_at) VALUES (?,?,?,?,?,?)")
      .run(id, data.learning, data.building, data.reading, data.thinking, now);
    return getDailyStatus()!;
  }
  memDailyStatus = { ...data, id: genId(), updatedAt: new Date().toISOString() };
  return memDailyStatus;
}

// ─── Learning Activity ────────────────────────────────

export function getLearningActivities(): LearningActivity[] {
  if (dbAvailable) {
    return db.prepare("SELECT * FROM learning_activity ORDER BY date ASC").all() as LearningActivity[];
  }
  return memActivities;
}

export function logLearningActivity(date: string, count: number = 1): void {
  if (dbAvailable) {
    db.prepare("INSERT INTO learning_activity (id, date, count) VALUES (?, ?, ?) ON CONFLICT(date) DO UPDATE SET count = count + ?")
      .run(genId(), date, count, count);
    return;
  }
  const existing = memActivities.find(a => a.date === date);
  if (existing) existing.count += count;
  else memActivities.push({ id: genId(), date, count });
}

// ─── Garden Entries ───────────────────────────────────

interface GardenRow { id: string; title: string; slug: string; content: string; excerpt: string; tags: string; category: string; stage: string; published: number; created_at: string; updated_at: string; }

export function getGardenEntries(): GardenEntry[] {
  if (dbAvailable) {
    return (db.prepare("SELECT * FROM garden_entries WHERE published = 1 ORDER BY created_at DESC").all() as GardenRow[]).map(r => ({
      id: r.id, title: r.title, slug: r.slug, content: r.content, excerpt: r.excerpt,
      tags: JSON.parse(r.tags || "[]"), category: r.category, stage: r.stage,
      published: r.published === 1, createdAt: r.created_at, updatedAt: r.updated_at,
    }));
  }
  return memGarden.filter(g => g.published);
}

export function getGardenEntryBySlug(slug: string): GardenEntry | null {
  if (dbAvailable) {
    const r = db.prepare("SELECT * FROM garden_entries WHERE slug = ? AND published = 1").get(slug) as GardenRow | undefined;
    return r ? {
      id: r.id, title: r.title, slug: r.slug, content: r.content, excerpt: r.excerpt,
      tags: JSON.parse(r.tags || "[]"), category: r.category, stage: r.stage,
      published: r.published === 1, createdAt: r.created_at, updatedAt: r.updated_at,
    } : null;
  }
  return memGarden.find(g => g.slug === slug && g.published) || null;
}

export function getGardenEntryById(id: string): GardenEntry | null {
  if (dbAvailable) {
    const r = db.prepare("SELECT * FROM garden_entries WHERE id = ?").get(id) as GardenRow | undefined;
    return r ? {
      id: r.id, title: r.title, slug: r.slug, content: r.content, excerpt: r.excerpt,
      tags: JSON.parse(r.tags || "[]"), category: r.category, stage: r.stage,
      published: r.published === 1, createdAt: r.created_at, updatedAt: r.updated_at,
    } : null;
  }
  return memGarden.find(g => g.id === id) || null;
}

export function createGardenEntry(data: Omit<GardenEntry, "id" | "createdAt" | "updatedAt">): GardenEntry {
  if (dbAvailable) {
    const id = genId(); const now = new Date().toISOString();
    db.prepare("INSERT INTO garden_entries (id,title,slug,content,excerpt,tags,category,stage,published,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)")
      .run(id, data.title, data.slug, data.content, data.excerpt, JSON.stringify(data.tags), data.category, data.stage, data.published ? 1 : 0, now, now);
    return getGardenEntryById(id)!;
  }
  const g: MemGarden = { ...data, id: genId(), createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() };
  memGarden.push(g);
  return g;
}

export function updateGardenEntry(id: string, data: Partial<Omit<GardenEntry, "id" | "createdAt">>): GardenEntry | null {
  if (dbAvailable) {
    const e = getGardenEntryById(id); if (!e) return null;
    const now = new Date().toISOString();
    db.prepare("UPDATE garden_entries SET title=?,slug=?,content=?,excerpt=?,tags=?,category=?,stage=?,published=?,updated_at=? WHERE id=?")
      .run(data.title ?? e.title, data.slug ?? e.slug, data.content ?? e.content, data.excerpt ?? e.excerpt, JSON.stringify(data.tags ?? e.tags), data.category ?? e.category, data.stage ?? e.stage, data.published !== undefined ? (data.published ? 1 : 0) : (e.published ? 1 : 0), now, id);
    return getGardenEntryById(id);
  }
  const idx = memGarden.findIndex(g => g.id === id);
  if (idx < 0) return null;
  Object.assign(memGarden[idx], data);
  memGarden[idx].updatedAt = new Date().toISOString();
  return memGarden[idx];
}

export function deleteGardenEntry(id: string): boolean {
  if (dbAvailable) return db.prepare("DELETE FROM garden_entries WHERE id = ?").run(id).changes > 0;
  const idx = memGarden.findIndex(g => g.id === id);
  if (idx >= 0) { memGarden.splice(idx, 1); return true; }
  return false;
}

// Initialize on first import
if (dbAvailable) {
  initTables();
}
