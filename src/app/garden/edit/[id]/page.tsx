import { redirect } from "next/navigation";
import { isAuthenticated } from "@/lib/auth";
import { getGardenEntryById } from "@/lib/db";
import { GardenEditorForm } from "../GardenEditorForm";

export default async function EditGardenPage({ params }: { params: Promise<{ id: string }> }) {
  if (!(await isAuthenticated())) redirect("/");
  const { id } = await params;
  const entry = getGardenEntryById(id);
  if (!entry) redirect("/garden");
  return <GardenEditorForm entry={entry} />;
}
