import { NextRequest, NextResponse } from "next/server";
import { isAuthenticated } from "@/lib/auth";
import { getProjects, createProject } from "@/lib/db";

export async function GET() {
  return NextResponse.json(getProjects());
}

export async function POST(req: NextRequest) {
  if (!(await isAuthenticated())) return NextResponse.json({ error: "未授权" }, { status: 401 });
  const data = await req.json();
  const project = createProject({
    title: data.title, category: data.category || "", description: data.description || "",
    tags: data.tags || [], featured: data.featured ?? false, sortOrder: data.sortOrder ?? 0,
  });
  return NextResponse.json(project, { status: 201 });
}
