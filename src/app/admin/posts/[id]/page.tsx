import { notFound } from "next/navigation";
import { getPostById } from "@/lib/db";
import PostEditor from "../PostEditor";

export default async function EditPostPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const post = getPostById(id);
  if (!post) notFound();

  return <PostEditor post={post} />;
}
