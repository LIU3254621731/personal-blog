import { redirect } from "next/navigation";
import { isAuthenticated } from "@/lib/auth";
import { getProjectById } from "@/lib/db";
import { ProjectEditorForm } from "../ProjectEditorForm";

export default async function EditProjectPage({ params }: { params: Promise<{ id: string }> }) {
  if (!(await isAuthenticated())) redirect("/");
  const { id } = await params;
  const project = getProjectById(id);
  if (!project) redirect("/projects");
  return <ProjectEditorForm project={project} />;
}
