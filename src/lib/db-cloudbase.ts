/**
 * CloudBase persistence — mirrors db.ts memory stores.
 *
 * When TCB_ENV is set:
 *   1. On first access, loads all data from CloudBase into memory
 *   2. All reads stay sync (from memory)
 *   3. All writes go to memory + CloudBase (async background)
 *
 * This keeps the existing synchronous db.ts interface intact while
 * adding CloudBase as a persistent backend for EdgeOne deployments.
 */

const TCB_ENV = process.env.TCB_ENV || process.env.TCB_ENV_ID || "";

let _app: any = null;
let _inited = false;

function getApp() {
  if (!TCB_ENV) return null;
  if (!_app && !_inited) {
    _inited = true;
    try {
      const tcb = require("@cloudbase/node-sdk");
      const opts: any = { env: TCB_ENV };
      if (process.env.TCB_SECRET_ID) {
        opts.secretId = process.env.TCB_SECRET_ID;
        opts.secretKey = process.env.TCB_SECRET_KEY;
      }
      _app = tcb.init(opts);
      console.log("[tcb] connected:", TCB_ENV);
    } catch (e: any) {
      console.warn("[tcb] init failed:", e.message?.substring(0, 100));
    }
  }
  return _app;
}

function db() { return getApp()?.database(); }
function coll(name: string) { return db()?.collection(name); }

export function isCloudBaseAvailable(): boolean {
  return !!getApp();
}

// ─── Load all data into memory stores ─────────────────

export async function loadAll(): Promise<{
  posts: any[]; projects: any[]; garden: any[]; siteConfig: Record<string, string>;
  dailyStatus: any; activities: any[];
}> {
  const empty = { posts: [], projects: [], garden: [], siteConfig: {}, dailyStatus: null, activities: [] as any[] };
  if (!isCloudBaseAvailable()) return empty;

  const d = db()!;
  try {
    const [posts, projects, garden, config, daily, activities] = await Promise.all([
      d.collection("posts").orderBy("createdAt", "desc").limit(200).get().then((r: any) => (r.data || [])).catch(() => []),
      d.collection("projects").orderBy("sortOrder", "asc").limit(100).get().then((r: any) => (r.data || [])).catch(() => []),
      d.collection("garden_entries").where({ published: true }).orderBy("createdAt", "desc").limit(100).get().then((r: any) => (r.data || [])).catch(() => []),
      d.collection("site_config").limit(200).get().then((r: any) => {
        const cfg: Record<string, string> = {};
        for (const row of r.data || []) cfg[row.key] = row.value;
        return cfg;
      }).catch(() => ({})),
      d.collection("daily_status").orderBy("updatedAt", "desc").limit(1).get().then((r: any) => (r.data?.[0] || null)).catch(() => null),
      d.collection("learning_activity").orderBy("date", "asc").limit(365).get().then((r: any) => (r.data || [])).catch(() => []),
    ]);

    console.log(`[tcb] loaded: ${posts.length} posts, ${projects.length} projects, ${garden.length} garden, ${Object.keys(config).length} config keys`);
    return { posts, projects, garden, siteConfig: config, dailyStatus: daily, activities };
  } catch (e: any) {
    console.warn("[tcb] loadAll failed:", e.message?.substring(0, 100));
    return empty;
  }
}

// ─── Write helpers (fire-and-forget to CloudBase) ─────

export async function savePost(post: any): Promise<void> {
  const c = coll("posts"); if (!c) return;
  await c.doc(post.id).set(post).catch(() => {});
}

export async function deletePostDoc(id: string): Promise<void> {
  const c = coll("posts"); if (!c) return;
  await c.doc(id).remove().catch(() => {});
}

export async function saveProject(proj: any): Promise<void> {
  const c = coll("projects"); if (!c) return;
  await c.doc(proj.id).set(proj).catch(() => {});
}

export async function deleteProjectDoc(id: string): Promise<void> {
  const c = coll("projects"); if (!c) return;
  await c.doc(id).remove().catch(() => {});
}

export async function saveSiteConfigKV(key: string, value: string): Promise<void> {
  const c = coll("site_config"); if (!c) return;
  const existing = await c.where({ key }).limit(1).get();
  if (existing.data?.length) {
    await c.doc(existing.data[0]._id).update({ value }).catch(() => {});
  } else {
    await c.add({ key, value }).catch(() => {});
  }
}

export async function deleteSiteConfigKV(key: string): Promise<void> {
  const c = coll("site_config"); if (!c) return;
  const existing = await c.where({ key }).limit(1).get();
  if (existing.data?.length) {
    await c.doc(existing.data[0]._id).remove().catch(() => {});
  }
}

export async function saveDailyStatus(status: any): Promise<void> {
  const c = coll("daily_status"); if (!c) return;
  await c.doc(status.id).set(status).catch(() => {});
}

export async function saveActivity(activity: any): Promise<void> {
  const c = coll("learning_activity"); if (!c) return;
  const existing = await c.where({ date: activity.date }).limit(1).get();
  if (existing.data?.length) {
    await c.doc(existing.data[0]._id).update({ count: activity.count }).catch(() => {});
  } else {
    await c.add({ date: activity.date, count: activity.count }).catch(() => {});
  }
}

export async function saveGardenEntry(entry: any): Promise<void> {
  const c = coll("garden_entries"); if (!c) return;
  await c.doc(entry.id).set(entry).catch(() => {});
}

export async function deleteGardenDoc(id: string): Promise<void> {
  const c = coll("garden_entries"); if (!c) return;
  await c.doc(id).remove().catch(() => {});
}
