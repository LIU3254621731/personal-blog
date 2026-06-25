import { NextRequest, NextResponse } from "next/server";
import { isAuthenticated } from "@/lib/auth";
import { getGardenEntryById, updateGardenEntry, deleteGardenEntry } from "@/lib/db";
import { validateBody, updateGardenSchema } from "@/lib/validation";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const entry = getGardenEntryById(id);
  if (!entry) return NextResponse.json({ error: "花园条目不存在" }, { status: 404 });
  return NextResponse.json(entry);
}

export async function PUT(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "未授权" }, { status: 401 });
  }

  const { id } = await params;
  const parsed = await validateBody(req, updateGardenSchema);
  if (parsed instanceof NextResponse) return parsed;

  const entry = updateGardenEntry(id, parsed);
  if (!entry) return NextResponse.json({ error: "花园条目不存在" }, { status: 404 });
  return NextResponse.json(entry);
}

export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "未授权" }, { status: 401 });
  }

  const { id } = await params;
  const ok = deleteGardenEntry(id);
  if (!ok) return NextResponse.json({ error: "花园条目不存在" }, { status: 404 });
  return NextResponse.json({ success: true });
}
