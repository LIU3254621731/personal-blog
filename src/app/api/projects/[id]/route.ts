import { NextRequest, NextResponse } from "next/server";
import { isAuthenticated } from "@/lib/auth";
import { getProjectById, updateProject, deleteProject } from "@/lib/db";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const project = getProjectById(id);
  if (!project)
    return NextResponse.json({ error: "项目不存在" }, { status: 404 });
  return NextResponse.json(project);
}

export async function PUT(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  if (!(await isAuthenticated()))
    return NextResponse.json({ error: "未授权" }, { status: 401 });

  const { id } = await params;
  const data = await req.json();
  const project = updateProject(id, {
    title: data.title,
    category: data.category,
    description: data.description,
    tags: data.tags,
    featured: data.featured,
    sortOrder: data.sortOrder,
  });
  if (!project)
    return NextResponse.json({ error: "项目不存在" }, { status: 404 });
  return NextResponse.json(project);
}

export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  if (!(await isAuthenticated()))
    return NextResponse.json({ error: "未授权" }, { status: 401 });

  const { id } = await params;
  const ok = deleteProject(id);
  if (!ok) return NextResponse.json({ error: "项目不存在" }, { status: 404 });
  return NextResponse.json({ success: true });
}
