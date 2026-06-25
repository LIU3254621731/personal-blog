import { redirect } from "next/navigation";
import { isAuthenticated } from "@/lib/auth";
import type { ReactNode } from "react";
import { AdminSidebar } from "./AdminSidebar";

export default async function AdminLayout({ children }: { children: ReactNode }) {
  const authed = await isAuthenticated();

  // Not authenticated — redirect to home. The admin login modal
  // can be opened from the footer "·" button on any public page.
  if (!authed) redirect("/");

  return (
    <div className="flex min-h-screen bg-bg-primary">
      <AdminSidebar />
      <main className="flex-1 p-8 lg:p-10 max-w-5xl mx-auto w-full">
        {children}
      </main>
    </div>
  );
}
