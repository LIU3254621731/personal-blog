// Script to apply all db.ts changes cleanly
const fs = require("fs");
const path = require("path");
const dbPath = path.join(__dirname, "..", "src", "lib", "db.ts");
let f = fs.readFileSync(dbPath, "utf8");

// 1. Add import after the top comment block
f = f.replace(
  " */\n\n// ─── Try loading SQLite",
  " */\n\nimport { seedMemStores } from \"./seed\";\n\n// ─── Try loading SQLite"
);

// 2. Add learning_paths table in initTables()
f = f.replace(
  "db.exec(`CREATE TABLE IF NOT EXISTS garden_entries (id TEXT PRIMARY KEY, title TEXT NOT NULL, slug TEXT UNIQUE NOT NULL, content TEXT DEFAULT '', excerpt TEXT DEFAULT '', tags TEXT DEFAULT '[]', category TEXT DEFAULT 'thought', stage TEXT DEFAULT 'seedling', published INTEGER DEFAULT 1, created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now')))`);\n}",
  "db.exec(`CREATE TABLE IF NOT EXISTS garden_entries (id TEXT PRIMARY KEY, title TEXT NOT NULL, slug TEXT UNIQUE NOT NULL, content TEXT DEFAULT '', excerpt TEXT DEFAULT '', tags TEXT DEFAULT '[]', category TEXT DEFAULT 'thought', stage TEXT DEFAULT 'seedling', published INTEGER DEFAULT 1, created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now')))`);\n  db.exec(`CREATE TABLE IF NOT EXISTS learning_paths (id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '未命名', description TEXT DEFAULT '', icon TEXT DEFAULT 'book', tags TEXT DEFAULT '[]', \"order\" INTEGER DEFAULT 99, created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now')))`);\n}"
);

// 3. Add MemLearningPath interface after MemGarden
f = f.replace(
  "interface MemGarden {\n  id: string; title: string; slug: string; content: string; excerpt: string;\n  tags: string[]; category: string; stage: string; published: boolean;\n  createdAt: string; updatedAt: string;\n}",
  "interface MemGarden {\n  id: string; title: string; slug: string; content: string; excerpt: string;\n  tags: string[]; category: string; stage: string; published: boolean;\n  createdAt: string; updatedAt: string;\n}\ninterface MemLearningPath {\n  id: string; name: string; description: string; icon: string;\n  tags: string[]; order: number; createdAt: string; updatedAt: string;\n}"
);

// 4. Add memLearningPaths after memGarden
f = f.replace(
  "let memGarden: MemGarden[] = [];\nlet memSiteConfig",
  "let memGarden: MemGarden[] = [];\nlet memLearningPaths: MemLearningPath[] = [];\nlet memSiteConfig"
);

// 5. Add ensureSeed function after memActivities
f = f.replace(
  "let memActivities: { id: string; date: string; count: number }[] = [];\n\n// ─── SQLite helpers",
  "let memActivities: { id: string; date: string; count: number }[] = [];\n\n// ─── Seed fallback (EdgeOne / no-SQLite environments) ─\n\nlet _seeded = false;\nfunction ensureSeed() {\n  if (!dbAvailable && !_seeded) {\n    _seeded = true;\n    seedMemStores(memPosts, memProjects, memGarden, memLearningPaths, memSiteConfig, memDailyStatus, memActivities);\n  }\n}\n\n// ─── SQLite helpers"
);

// 6. Add LearningPath export interface after GardenEntry
f = f.replace(
  "export interface GardenEntry {\n  id: string; title: string; slug: string; content: string; excerpt: string;\n  tags: string[]; category: string; stage: string; published: boolean;\n  createdAt: string; updatedAt: string;\n}",
  "export interface GardenEntry {\n  id: string; title: string; slug: string; content: string; excerpt: string;\n  tags: string[]; category: string; stage: string; published: boolean;\n  createdAt: string; updatedAt: string;\n}\nexport interface LearningPath {\n  id: string; name: string; description: string; icon: string;\n  tags: string[]; order: number; createdAt: string; updatedAt: string;\n}"
);

// 7. Add ensureSeed() before each mem* return in fallback getters
// getPosts
f = f.replace("  }\n  return memPosts;", "  }\n  ensureSeed();\n  return memPosts;");
// getProjects
f = f.replace("  }\n  return memProjects;", "  }\n  ensureSeed();\n  return memProjects;");
// getSiteConfig
f = f.replace("  }\n  return { ...memSiteConfig };", "  }\n  ensureSeed();\n  return { ...memSiteConfig };");
// getDailyStatus
f = f.replace("  }\n  return memDailyStatus;", "  }\n  ensureSeed();\n  return memDailyStatus;");
// getLearningActivities
f = f.replace("  }\n  return memActivities;", "  }\n  ensureSeed();\n  return memActivities;");
// getGardenEntries (return memGarden.filter)
f = f.replace("  }\n  return memGarden.filter", "  }\n  ensureSeed();\n  return memGarden.filter");
// getGardenEntryBySlug (return memGarden.find)
// The first one after getGardenEntries will match getGardenEntryBySlug
f = f.replace("  }\n  return memGarden.find(g => g.slug", "  }\n  ensureSeed();\n  return memGarden.find(g => g.slug");
// getGardenEntryById
f = f.replace("  }\n  return memGarden.find(g => g.id", "  }\n  ensureSeed();\n  return memGarden.find(g => g.id");

// 8. Add learning_paths CRUD functions + migration before "Initialize on first import"
const learningPathsCode = `
// ─── Learning Paths ──────────────────────────────────

interface LearningPathRow { id: string; name: string; description: string; icon: string; tags: string; order: number; created_at: string; updated_at: string; }

/** Migrate legacy learning_path_* keys from site_config to learning_paths table. */
function migrateLearningPathsFromSiteConfig(): void {
  if (!dbAvailable) return;
  const existing = db.prepare("SELECT COUNT(*) AS c FROM learning_paths").get() as { c: number };
  if (existing.c > 0) return;
  const rows = db.prepare("SELECT key, value FROM site_config WHERE key LIKE 'learning_path_%'").all() as { key: string; value: string }[];
  if (rows.length === 0) return;
  const insert = db.prepare("INSERT OR REPLACE INTO learning_paths (id, name, description, icon, tags, \\"order\\", created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)");
  const t = db.transaction(() => {
    for (const r of rows) {
      const id = r.key.replace("learning_path_", "");
      try {
        const d = JSON.parse(r.value);
        insert.run(id, d.name || "未命名", d.description || "", d.icon || "book", JSON.stringify(d.tags || []), d.order ?? 99, new Date().toISOString(), new Date().toISOString());
      } catch {
        insert.run(id, r.value, "", "book", "[]", 99, new Date().toISOString(), new Date().toISOString());
      }
    }
    db.prepare("DELETE FROM site_config WHERE key LIKE 'learning_path_%'").run();
  });
  t();
}

export function getLearningPaths(): LearningPath[] {
  if (dbAvailable) {
    migrateLearningPathsFromSiteConfig();
    return (db.prepare("SELECT * FROM learning_paths ORDER BY \\"order\\" ASC, created_at DESC").all() as LearningPathRow[]).map(r => ({
      id: r.id, name: r.name, description: r.description, icon: r.icon,
      tags: JSON.parse(r.tags || "[]"), order: r.order,
      createdAt: r.created_at, updatedAt: r.updated_at,
    }));
  }
  ensureSeed();
  return memLearningPaths.sort((a, b) => (a.order ?? 99) - (b.order ?? 99));
}

export function getLearningPathById(id: string): LearningPath | null {
  if (dbAvailable) {
    migrateLearningPathsFromSiteConfig();
    const r = db.prepare("SELECT * FROM learning_paths WHERE id = ?").get(id) as LearningPathRow | undefined;
    return r ? {
      id: r.id, name: r.name, description: r.description, icon: r.icon,
      tags: JSON.parse(r.tags || "[]"), order: r.order,
      createdAt: r.created_at, updatedAt: r.updated_at,
    } : null;
  }
  ensureSeed();
  return memLearningPaths.find(p => p.id === id) || null;
}

export function createLearningPath(data: Omit<LearningPath, "createdAt" | "updatedAt">): LearningPath {
  if (dbAvailable) {
    const now = new Date().toISOString();
    db.prepare("INSERT INTO learning_paths (id, name, description, icon, tags, \\"order\\", created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)")
      .run(data.id, data.name, data.description, data.icon, JSON.stringify(data.tags), data.order, now, now);
    return getLearningPathById(data.id)!;
  }
  const p: MemLearningPath = { ...data, createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() };
  memLearningPaths.push(p);
  return p;
}

export function updateLearningPath(id: string, data: Partial<Omit<LearningPath, "id" | "createdAt">>): LearningPath | null {
  if (dbAvailable) {
    const e = getLearningPathById(id); if (!e) return null;
    const now = new Date().toISOString();
    db.prepare("UPDATE learning_paths SET name=?,description=?,icon=?,tags=?,\\"order\\"=?,updated_at=? WHERE id=?")
      .run(data.name ?? e.name, data.description ?? e.description, data.icon ?? e.icon, JSON.stringify(data.tags ?? e.tags), data.order ?? e.order, now, id);
    return getLearningPathById(id);
  }
  const idx = memLearningPaths.findIndex(p => p.id === id);
  if (idx < 0) return null;
  Object.assign(memLearningPaths[idx], data);
  memLearningPaths[idx].updatedAt = new Date().toISOString();
  return memLearningPaths[idx];
}

export function deleteLearningPath(id: string): boolean {
  if (dbAvailable) return db.prepare("DELETE FROM learning_paths WHERE id = ?").run(id).changes > 0;
  const idx = memLearningPaths.findIndex(p => p.id === id);
  if (idx >= 0) { memLearningPaths.splice(idx, 1); return true; }
  return false;
}

`;

f = f.replace(
  "\n// Initialize on first import",
  learningPathsCode + "// Initialize on first import"
);

fs.writeFileSync(dbPath, f, "utf8");
console.log("Done. All changes applied to db.ts");
