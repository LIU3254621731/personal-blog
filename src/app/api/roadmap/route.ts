import { NextRequest, NextResponse } from "next/server";
import { isAuthenticated } from "@/lib/auth";
import { getSiteConfig, setSiteConfig } from "@/lib/db";
import { v4 as uuid } from "uuid";

const KEY_PREFIX = "roadmap_";

export async function GET() {
  const config = getSiteConfig();
  const items: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(config)) {
    if (k.startsWith(KEY_PREFIX)) {
      try {
        items[k] = JSON.parse(v);
      } catch {
        items[k] = v;
      }
    }
  }
  return NextResponse.json(items);
}

export async function POST(req: NextRequest) {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "未授权" }, { status: 401 });
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "请求体必须是有效的 JSON" }, { status: 400 });
  }

  const key = `${KEY_PREFIX}${uuid()}`;
  setSiteConfig(key, JSON.stringify(body));
  return NextResponse.json({ key, value: body }, { status: 201 });
}
