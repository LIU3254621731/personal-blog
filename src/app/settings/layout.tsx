import { redirect } from "next/navigation";
import { isAuthenticated } from "@/lib/auth";

export default async function SettingsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  if (!(await isAuthenticated())) redirect("/");
  return <>{children}</>;
}
