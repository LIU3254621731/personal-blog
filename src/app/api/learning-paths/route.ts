import { NextRequest, NextResponse } from "next/server";
import { isAuthenticated } from "@/lib/auth";
import { getSiteConfig, setSiteConfig, deleteSiteConfig } from "@/lib/db";

// GET: list all learning paths
export async function GET() {
  const config = getSiteConfig();
  const paths: any[] = [];
  for (const [key, value] of Object.entries(config)) {
    if (!key.startsWith("learning_path_")) continue;
    try {
      paths.push({ id: key.replace("learning_path_", ""), ...JSON.parse(value) });
    } catch { /* skip */ }
  }
  paths.sort((a, b) => (a.order ?? 99) - (b.order ?? 99));
  return NextResponse.json(paths);
}

// POST: create new learning path
export async function POST(req: NextRequest) {
  if (!(await isAuthenticated())) return NextResponse.json({ error: "未授权" }, { status: 401 });
  try {
    const body = await req.json();
    const id = body.id || `path_${Date.now().toString(36)}`;
    const key = `learning_path_${id}`;
    setSiteConfig(key, JSON.stringify({
      name: body.name || "未命名",
      description: body.description || "",
      icon: body.icon || "book",
      tags: body.tags || [],
      order: body.order ?? 99,
    }));
    return NextResponse.json({ id, ...JSON.parse(getSiteConfig()[key]) }, { status: 201 });
  } catch {
    return NextResponse.json({ error: "数据格式错误" }, { status: 400 });
  }
}
