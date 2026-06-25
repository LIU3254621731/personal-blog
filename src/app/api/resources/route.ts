import { NextRequest, NextResponse } from "next/server";
import { isAuthenticated } from "@/lib/auth";
import { getSiteConfig, setSiteConfig } from "@/lib/db";
import { v4 as uuid } from "uuid";

const KEY_PREFIX = "resource_";

export async function GET() {
  const config = getSiteConfig();
  const items: Record<string, string> = {};
  for (const [k, v] of Object.entries(config)) {
    if (k.startsWith(KEY_PREFIX)) items[k] = v;
  }
  return NextResponse.json(items);
}

export async function POST(req: NextRequest) {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "未授权" }, { status: 401 });
  }
  let body: any;
  try { body = await req.json(); } catch {
    return NextResponse.json({ error: "请求体无效" }, { status: 400 });
  }
  const key = `${KEY_PREFIX}${uuid()}`;
  const { key: _k, ...value } = body;
  setSiteConfig(key, JSON.stringify(value));
  return NextResponse.json({ key, ...value }, { status: 201 });
}
