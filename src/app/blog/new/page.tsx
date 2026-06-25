import { redirect } from "next/navigation";
import { isAuthenticated } from "@/lib/auth";
import { PostEditorForm } from "../edit/PostEditorForm";

export default async function NewPostPage() {
  if (!(await isAuthenticated())) redirect("/");
  return <PostEditorForm />;
}
