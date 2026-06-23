import { NextRequest, NextResponse } from "next/server";
import { isAuthenticated } from "@/lib/auth";
import { getPostById, updatePost, deletePost } from "@/lib/db";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const post = getPostById(id);
  if (!post) return NextResponse.json({ error: "文章不存在" }, { status: 404 });
  return NextResponse.json(post);
}

export async function PUT(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  if (!(await isAuthenticated()))
    return NextResponse.json({ error: "未授权" }, { status: 401 });

  const { id } = await params;
  const data = await req.json();
  const post = updatePost(id, {
    title: data.title,
    slug: data.slug,
    content: data.content,
    excerpt: data.excerpt,
    tags: data.tags,
    published: data.published,
  });
  if (!post) return NextResponse.json({ error: "文章不存在" }, { status: 404 });
  return NextResponse.json(post);
}

export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  if (!(await isAuthenticated()))
    return NextResponse.json({ error: "未授权" }, { status: 401 });

  const { id } = await params;
  const ok = deletePost(id);
  if (!ok) return NextResponse.json({ error: "文章不存在" }, { status: 404 });
  return NextResponse.json({ success: true });
}
