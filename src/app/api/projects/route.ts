import { NextRequest, NextResponse } from "next/server";
import { isAuthenticated } from "@/lib/auth";
import { getProjects, createProject } from "@/lib/db";
import { validateBody, createProjectSchema } from "@/lib/validation";

export async function GET() {
  return NextResponse.json(getProjects());
}

export async function POST(req: NextRequest) {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "未授权" }, { status: 401 });
  }

  const parsed = await validateBody(req, createProjectSchema);
  if (parsed instanceof NextResponse) return parsed;

  const project = createProject(parsed);
  return NextResponse.json(project, { status: 201 });
}
