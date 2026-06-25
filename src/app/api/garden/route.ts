import { NextRequest, NextResponse } from "next/server";
import { isAuthenticated, verifyCsrf } from "@/lib/auth";
import { getGardenEntries, createGardenEntry } from "@/lib/db";
import { validateBody, createGardenSchema } from "@/lib/validation";

export async function GET() {
  return NextResponse.json(getGardenEntries());
}

export async function POST(req: NextRequest) {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "未授权" }, { status: 401 });
  }
  if (!(await verifyCsrf(req))) {
    return NextResponse.json({ error: "CSRF 校验失败" }, { status: 403 });
  }

  const parsed = await validateBody(req, createGardenSchema);
  if (parsed instanceof NextResponse) return parsed;

  const entry = createGardenEntry(parsed);
  return NextResponse.json(entry, { status: 201 });
}
