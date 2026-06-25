import { redirect } from "next/navigation";
import { isAuthenticated } from "@/lib/auth";
import { getPostBySlug, getPosts } from "@/lib/db";
import { PostEditorForm } from "../PostEditorForm";

export default async function EditPostPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  if (!(await isAuthenticated())) redirect("/");
  const { slug } = await params;
  const post = getPostBySlug(slug);
  if (!post) redirect("/blog");
  return <PostEditorForm post={post} />;
}

// Generate static params (empty since this page is dynamic per post)
export function generateStaticParams() {
  return [];
}
