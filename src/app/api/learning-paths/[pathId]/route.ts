import { NextRequest, NextResponse } from "next/server";
import { isAuthenticated } from "@/lib/auth";
import { getSiteConfig, setSiteConfig, deleteSiteConfig } from "@/lib/db";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ pathId: string }> },
) {
  const { pathId } = await params;
  const config = getSiteConfig();
  const key = `learning_path_${pathId}`;
  const raw = config[key];
  if (!raw) return NextResponse.json({ error: "路线不存在" }, { status: 404 });
  try {
    return NextResponse.json({ id: pathId, ...JSON.parse(raw) });
  } catch {
    return NextResponse.json({ id: pathId, name: raw });
  }
}

export async function PUT(
  req: NextRequest,
  { params }: { params: Promise<{ pathId: string }> },
) {
  if (!(await isAuthenticated())) return NextResponse.json({ error: "未授权" }, { status: 401 });
  const { pathId } = await params;
  const key = `learning_path_${pathId}`;
  const config = getSiteConfig();
  if (!config[key]) return NextResponse.json({ error: "路线不存在" }, { status: 404 });
  try {
    const existing = JSON.parse(config[key]);
    const body = await req.json();
    const updated = { ...existing, ...body };
    setSiteConfig(key, JSON.stringify(updated));
    return NextResponse.json({ id: pathId, ...updated });
  } catch {
    return NextResponse.json({ error: "数据格式错误" }, { status: 400 });
  }
}

export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ pathId: string }> },
) {
  if (!(await isAuthenticated())) return NextResponse.json({ error: "未授权" }, { status: 401 });
  const { pathId } = await params;
  deleteSiteConfig(`learning_path_${pathId}`);
  return NextResponse.json({ success: true });
}
