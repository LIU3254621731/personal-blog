import { NextRequest, NextResponse } from "next/server";
import { isAuthenticated, verifyCsrf } from "@/lib/auth";
import { getPosts, createPost, logLearningActivity } from "@/lib/db";
import { validateBody, createPostSchema } from "@/lib/validation";

export async function GET() {
  return NextResponse.json(getPosts());
}

export async function POST(req: NextRequest) {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "未授权" }, { status: 401 });
  }
  if (!(await verifyCsrf(req))) {
    return NextResponse.json({ error: "CSRF 校验失败" }, { status: 403 });
  }

  const parsed = await validateBody(req, createPostSchema);
  if (parsed instanceof NextResponse) return parsed;

  const post = createPost(parsed);
  return NextResponse.json(post, { status: 201 });
}
