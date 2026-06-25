import { redirect } from "next/navigation";
import { isAuthenticated } from "@/lib/auth";
import { GardenEditorForm } from "../edit/GardenEditorForm";

export default async function NewGardenPage() {
  if (!(await isAuthenticated())) redirect("/");
  return <GardenEditorForm />;
}
