/**
 * Fetch CloudBase data at build time and write to a JSON file.
 *
 * Called as a prebuild step:  node scripts/fetch-cloudbase-data.cjs
 * Writes to:  data/cloudbase-data.json
 *
 * Requires env vars: TCB_ENV_ID, TCB_SECRET_ID, TCB_SECRET_KEY
 */

const fs = require("fs");
const path = require("path");

// Load .env for local dev
const envPath = path.join(__dirname, "..", ".env");
if (fs.existsSync(envPath)) {
  for (const line of fs.readFileSync(envPath, "utf8").split("\n")) {
    const m = line.match(/^([A-Z_]+)="?([^"]+)"?$/);
    if (m) process.env[m[1]] = m[2];
  }
}

const TCB_ENV = process.env.TCB_ENV_ID || process.env.TCB_ENV || "";
const TCB_SID = process.env.TCB_SECRET_ID || "";
const TCB_SKEY = process.env.TCB_SECRET_KEY || "";

async function main() {
  if (!TCB_ENV) {
    console.log("[fetch-cloudbase] TCB_ENV_ID not set, skipping");
    return;
  }

  console.log("[fetch-cloudbase] connecting to:", TCB_ENV);

  const tcb = require("@cloudbase/node-sdk");
  const app = tcb.init({
    env: TCB_ENV,
    secretId: TCB_SID || undefined,
    secretKey: TCB_SKEY || undefined,
  });
  const db = app.database();

  async function fetchAll(name, query) {
    try {
      const res = await query;
      console.log(`[fetch-cloudbase] ${name}: ${(res.data || []).length} records`);
      return res.data || [];
    } catch (e) {
      console.warn(`[fetch-cloudbase] ${name} failed:`, e.message?.substring(0, 80));
      return [];
    }
  }

  const [posts, projects, siteConfig, garden, dailyStatus, activities] = await Promise.all([
    fetchAll("posts", db.collection("posts").orderBy("createdAt", "desc").limit(300).get()),
    fetchAll("projects", db.collection("projects").orderBy("sortOrder", "asc").limit(100).get()),
    fetchAll("site_config", db.collection("site_config").limit(200).get()),
    fetchAll("garden_entries", db.collection("garden_entries").where({ published: true }).orderBy("createdAt", "desc").limit(100).get()),
    fetchAll("daily_status", db.collection("daily_status").orderBy("updatedAt", "desc").limit(1).get()),
    fetchAll("learning_activity", db.collection("learning_activity").orderBy("date", "asc").limit(365).get()),
  ]);

  const data = {
    posts: posts.map(r => ({
      id: r._id || r.id, title: r.title, slug: r.slug, content: r.content || "",
      excerpt: r.excerpt || "", tags: r.tags || [],
      published: r.published !== false,
      createdAt: r.createdAt || r.created_at || "", updatedAt: r.updatedAt || r.updated_at || "",
    })),
    projects: projects.map(r => ({
      id: r._id || r.id, title: r.title, category: r.category || "",
      description: r.description || "", tags: r.tags || [],
      featured: r.featured === true, sortOrder: r.sortOrder ?? r.sort_order ?? 0,
      status: r.status || "building",
      githubUrl: r.githubUrl || r.github_url || "",
      demoUrl: r.demoUrl || r.demo_url || "",
      createdAt: r.createdAt || r.created_at || "", updatedAt: r.updatedAt || r.updated_at || "",
    })),
    siteConfig: (siteConfig || []).reduce((acc, r) => { acc[r.key] = r.value; return acc; }, {}),
    garden: garden.map(r => ({
      id: r._id || r.id, title: r.title, slug: r.slug, content: r.content || "",
      excerpt: r.excerpt || "", tags: r.tags || [],
      category: r.category || "thought", stage: r.stage || "seedling",
      published: r.published !== false,
      createdAt: r.createdAt || r.created_at || "", updatedAt: r.updatedAt || r.updated_at || "",
    })),
    dailyStatus: dailyStatus[0] ? {
      id: dailyStatus[0]._id || dailyStatus[0].id,
      learning: dailyStatus[0].learning || "",
      building: dailyStatus[0].building || "",
      reading: dailyStatus[0].reading || "",
      thinking: dailyStatus[0].thinking || "",
      updatedAt: dailyStatus[0].updatedAt || dailyStatus[0].updated_at || "",
    } : null,
    activities: activities.map(r => ({
      id: r._id || r.id, date: r.date, count: r.count || 0,
    })),
  };

  const outPath = path.join(__dirname, "..", "data", "cloudbase-data.json");
  const dir = path.dirname(outPath);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify(data), "utf8");

  const size = (fs.statSync(outPath).size / 1024).toFixed(1);
  console.log(`[fetch-cloudbase] written: ${outPath} (${size} KB)`);
  console.log(`[fetch-cloudbase] summary: ${data.posts.length} posts, ${data.projects.length} projects, ${Object.keys(data.siteConfig).length} config, ${data.garden.length} garden`);
}

main().catch(e => {
  console.error("[fetch-cloudbase] failed:", e.message);
  process.exit(1);
});
