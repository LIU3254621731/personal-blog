/** Seed CloudBase from local SQLite. Usage: node scripts/seed-cloudbase.cjs */
const path = require("path"); const fs = require("fs");
const envPath = path.join(__dirname, "..", ".env");
if (fs.existsSync(envPath)) {
  for (const line of fs.readFileSync(envPath, "utf8").split("\n")) {
    const m = line.match(/^([A-Z_]+)="?([^"]+)"?$/);
    if (m) process.env[m[1]] = m[2];
  }
}
const TCB_ENV = process.env.TCB_ENV_ID || process.env.TCB_ENV;
const TCB_SID = process.env.TCB_SECRET_ID; const TCB_SKEY = process.env.TCB_SECRET_KEY;
if (!TCB_ENV) { console.error("Missing TCB_ENV_ID"); process.exit(1); }

async function main() {
  const sqlite = require("better-sqlite3");
  const dbPath = path.join(__dirname, "..", "data", "blog.db");
  const db = new sqlite(dbPath);

  const posts = db.prepare("SELECT * FROM posts").all().map(r => ({
    _id: r.id, title: r.title, slug: r.slug, content: r.content,
    excerpt: r.excerpt, tags: JSON.parse(r.tags || "[]"),
    published: r.published === 1, createdAt: r.created_at, updatedAt: r.updated_at,
  }));
  const projects = db.prepare("SELECT * FROM projects").all().map(r => ({
    _id: r.id, title: r.title, category: r.category, description: r.description,
    tags: JSON.parse(r.tags || "[]"), featured: r.featured === 1,
    sortOrder: r.sort_order, status: r.status || "building",
    githubUrl: r.github_url || "", demoUrl: r.demo_url || "",
    createdAt: r.created_at, updatedAt: r.updated_at,
  }));
  const siteConfig = db.prepare("SELECT * FROM site_config").all().map(r => ({
    _id: r.key, key: r.key, value: r.value,
  }));
  const garden = db.prepare("SELECT * FROM garden_entries").all().map(r => ({
    _id: r.id, title: r.title, slug: r.slug, content: r.content,
    excerpt: r.excerpt, tags: JSON.parse(r.tags || "[]"),
    category: r.category, stage: r.stage,
    published: r.published === 1, createdAt: r.created_at, updatedAt: r.updated_at,
  }));
  const daily = db.prepare("SELECT * FROM daily_status ORDER BY updated_at DESC LIMIT 1").all();
  const activities = db.prepare("SELECT * FROM learning_activity").all();

  console.log(`Local: ${posts.length} posts, ${projects.length} projects, ${siteConfig.length} config, ${garden.length} garden, ${daily.length} daily, ${activities.length} activities`);

  const tcb = require("@cloudbase/node-sdk");
  const app = tcb.init({ env: TCB_ENV, secretId: TCB_SID, secretKey: TCB_SKEY });
  const tcbDb = app.database();

  async function seed(name, data) {
    if (data.length === 0) { console.log(`  ${name}: empty`); return; }
    const coll = tcbDb.collection(name);
    let created = 0, updated = 0, failed = 0;
    // Batch: 20 at a time
    const batchSize = 20;
    for (let i = 0; i < data.length; i += batchSize) {
      const batch = data.slice(i, i + batchSize);
      const results = await Promise.allSettled(batch.map(async (doc) => {
        const { _id, ...fields } = doc;
        try {
          // Try get first to see if it exists
          const existing = await coll.doc(_id).get();
          if (existing.data && existing.data.length > 0) {
            await coll.doc(_id).update(fields);
            return "updated";
          }
        } catch (e) {
          // Not found or other error — try set
        }
        try {
          await coll.doc(_id).set(fields);
          return "created";
        } catch (e) {
          console.error(`  ${name} ${_id}:`, e.message?.substring(0, 80));
          return "failed";
        }
      }));
      for (const r of results) {
        if (r.status === "rejected") failed++;
        else if (r.value === "created") created++;
        else if (r.value === "updated") updated++;
        else failed++;
      }
    }
    console.log(`  ${name}: ${created} created, ${updated} updated, ${failed} failed (of ${data.length})`);
  }

  await seed("posts", posts);
  await seed("projects", projects);
  await seed("site_config", siteConfig);
  await seed("garden_entries", garden);
  if (daily.length > 0) {
    await seed("daily_status", [{ _id: daily[0].id, learning: daily[0].learning, building: daily[0].building, reading: daily[0].reading, thinking: daily[0].thinking, updatedAt: daily[0].updated_at }]);
  }
  await seed("learning_activity", activities.map(r => ({ _id: String(r.id), date: r.date, count: r.count })));

  console.log("\nDone!");
}
main().catch(e => { console.error(e.message); process.exit(1); });
