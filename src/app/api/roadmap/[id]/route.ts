import { NextRequest, NextResponse } from "next/server";
import { isAuthenticated } from "@/lib/auth";
import { getSiteConfig, setSiteConfig, deleteSiteConfig } from "@/lib/db";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const config = getSiteConfig();
  const value = config[id];
  if (value === undefined) {
    return NextResponse.json({ error: "路线图条目不存在" }, { status: 404 });
  }
  try {
    return NextResponse.json({ key: id, value: JSON.parse(value) });
  } catch {
    return NextResponse.json({ key: id, value });
  }
}

export async function PUT(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "未授权" }, { status: 401 });
  }

  const { id } = await params;

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "请求体必须是有效的 JSON" }, { status: 400 });
  }

  // Check if the key exists
  const config = getSiteConfig();
  if (!(id in config)) {
    return NextResponse.json({ error: "路线图条目不存在" }, { status: 404 });
  }

  setSiteConfig(id, JSON.stringify(body));
  return NextResponse.json({ key: id, value: body });
}

export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "未授权" }, { status: 401 });
  }

  const { id } = await params;
  const ok = deleteSiteConfig(id);
  if (!ok) return NextResponse.json({ error: "路线图条目不存在" }, { status: 404 });
  return NextResponse.json({ success: true });
}
