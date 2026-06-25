import { redirect } from "next/navigation";
import { isAuthenticated } from "@/lib/auth";
import { ProjectEditorForm } from "../edit/ProjectEditorForm";

export default async function NewProjectPage() {
  if (!(await isAuthenticated())) redirect("/");
  return <ProjectEditorForm />;
}
