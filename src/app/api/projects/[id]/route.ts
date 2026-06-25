import { NextRequest, NextResponse } from "next/server";
import { isAuthenticated, verifyCsrf } from "@/lib/auth";
import { getProjectById, updateProject, deleteProject } from "@/lib/db";
import { validateBody, updateProjectSchema } from "@/lib/validation";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const project = getProjectById(id);
  if (!project) {
    return NextResponse.json({ error: "项目不存在" }, { status: 404 });
  }
  return NextResponse.json(project);
}

export async function PUT(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "未授权" }, { status: 401 });
  }
  if (!(await verifyCsrf(req))) {
    return NextResponse.json({ error: "CSRF 校验失败" }, { status: 403 });
  }

  const { id } = await params;
  const parsed = await validateBody(req, updateProjectSchema);
  if (parsed instanceof NextResponse) return parsed;

  const project = updateProject(id, parsed);
  if (!project) {
    return NextResponse.json({ error: "项目不存在" }, { status: 404 });
  }
  return NextResponse.json(project);
}

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "未授权" }, { status: 401 });
  }
  if (!(await verifyCsrf(req))) {
    return NextResponse.json({ error: "CSRF 校验失败" }, { status: 403 });
  }

  const { id } = await params;
  const ok = deleteProject(id);
  if (!ok) return NextResponse.json({ error: "项目不存在" }, { status: 404 });
  return NextResponse.json({ success: true });
}
