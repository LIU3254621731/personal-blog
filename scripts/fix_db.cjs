const fs = require("fs");
const path = require("path");
const dbPath = path.join(__dirname, "..", "src", "lib", "db.ts");
let f = fs.readFileSync(dbPath, "utf8");

// 1. Add import after the JSDoc comment block
f = f.replace(" */\n\n// ─── Try loading SQLite", " */\n\nimport { seedMemStores } from \"./seed\";\n\n// ─── Try loading SQLite");

// 2. Add MemLearningPath after MemGarden
f = f.replace("interface MemGarden {\n  id: string; title: string; slug: string; content: string; excerpt: string;\n  tags: string[]; category: string; stage: string; published: boolean;\n  createdAt: string; updatedAt: string;\n}", "interface MemGarden {\n  id: string; title: string; slug: string; content: string; excerpt: string;\n  tags: string[]; category: string; stage: string; published: boolean;\n  createdAt: string; updatedAt: string;\n}\ninterface MemLearningPath {\n  id: string; name: string; description: string; icon: string;\n  tags: string[]; order: number; createdAt: string; updatedAt: string;\n}");

// 3. Add memLearningPaths + ensureSeed after memGarden
f = f.replace("let memGarden: MemGarden[] = [];\nlet memSiteConfig: Record<string, string> = {};\nlet memDailyStatus: any = null;\nlet memActivities: { id: string; date: string; count: number }[] = [];\n\n// ─── SQLite helpers", "let memGarden: MemGarden[] = [];\nlet memLearningPaths: MemLearningPath[] = [];\nlet memSiteConfig: Record<string, string> = {};\nlet memDailyStatus: any = null;\nlet memActivities: { id: string; date: string; count: number }[] = [];\n\nlet _seeded = false;\nfunction ensureSeed() {\n  if (!dbAvailable && !_seeded) {\n    _seeded = true;\n    seedMemStores(memPosts, memProjects, memGarden, memLearningPaths, memSiteConfig, memDailyStatus, memActivities);\n  }\n}\n\n// ─── SQLite helpers");

// 4. Add learning_paths table to initTables
f = f.replace("db.exec(`CREATE TABLE IF NOT EXISTS garden_entries (id TEXT PRIMARY KEY, title TEXT NOT NULL, slug TEXT UNIQUE NOT NULL, content TEXT DEFAULT '', excerpt TEXT DEFAULT '', tags TEXT DEFAULT '[]', category TEXT DEFAULT 'thought', stage TEXT DEFAULT 'seedling', published INTEGER DEFAULT 1, created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now')))`);\n}", "db.exec(`CREATE TABLE IF NOT EXISTS garden_entries (id TEXT PRIMARY KEY, title TEXT NOT NULL, slug TEXT UNIQUE NOT NULL, content TEXT DEFAULT '', excerpt TEXT DEFAULT '', tags TEXT DEFAULT '[]', category TEXT DEFAULT 'thought', stage TEXT DEFAULT 'seedling', published INTEGER DEFAULT 1, created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now')))`);\n  db.exec(`CREATE TABLE IF NOT EXISTS learning_paths (id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '未命名', description TEXT DEFAULT '', icon TEXT DEFAULT 'book', tags TEXT DEFAULT '[]', \"order\" INTEGER DEFAULT 99, created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now')))`);\n}");

// 5. Add LearningPath export interface after GardenEntry
f = f.replace("export interface GardenEntry {\n  id: string; title: string; slug: string; content: string; excerpt: string;\n  tags: string[]; category: string; stage: string; published: boolean;\n  createdAt: string; updatedAt: string;\n}\n\n// ─── Posts", "export interface GardenEntry {\n  id: string; title: string; slug: string; content: string; excerpt: string;\n  tags: string[]; category: string; stage: string; published: boolean;\n  createdAt: string; updatedAt: string;\n}\nexport interface LearningPath {\n  id: string; name: string; description: string; icon: string;\n  tags: string[]; order: number; createdAt: string; updatedAt: string;\n}\n\n// ─── Posts");

// 6. Add ensureSeed() in getters before return mem*
const addSeed = (funcName, storeName) => {
  const pat = new RegExp("(export function " + funcName + "[^}]+if \\(dbAvailable\\) \\{[^}]+\\})\\n  return " + storeName, "s");
  f = f.replace(pat, '$1\n  ensureSeed();\n  return ' + storeName);
};
addSeed("getPosts", "memPosts");
addSeed("getProjects", "memProjects");
addSeed("getGardenEntries", "memGarden.filter");

fs.writeFileSync(dbPath, f, "utf8");
console.log("Done - db.ts fixed");
