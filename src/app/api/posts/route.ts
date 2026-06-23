import { NextRequest, NextResponse } from "next/server";
import { isAuthenticated } from "@/lib/auth";
import { getPosts, createPost } from "@/lib/db";

export async function GET() {
  return NextResponse.json(getPosts());
}

export async function POST(req: NextRequest) {
  if (!(await isAuthenticated())) return NextResponse.json({ error: "未授权" }, { status: 401 });
  const data = await req.json();
  const post = createPost({
    title: data.title, slug: data.slug || data.title.toLowerCase().replace(/\s+/g, "-").replace(/[^a-z0-9一-鿿-]/g, "").slice(0, 60),
    content: data.content || "", excerpt: data.excerpt || "", tags: data.tags || [], published: data.published ?? true,
  });
  return NextResponse.json(post, { status: 201 });
}
