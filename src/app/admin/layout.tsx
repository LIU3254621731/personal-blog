import { redirect } from "next/navigation";
import { headers } from "next/headers";
import { isAuthenticated } from "@/lib/auth";
import Link from "next/link";
import type { ReactNode } from "react";
import { LogoutButton } from "./LogoutButton";

export default async function AdminLayout({ children }: { children: ReactNode }) {
  const authed = await isAuthenticated();
  const pathname = (await headers()).get("x-pathname") || "";
  // Skip auth check for login page to prevent redirect loop
  if (!authed && pathname !== "/admin/login") redirect("/admin/login");

  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>
      <aside style={{ width: 220, background: "#fff", borderRight: "1px solid #e8e6e1", padding: "24px 0", display: "flex", flexDirection: "column", flexShrink: 0 }}>
        <div style={{ padding: "0 20px 24px", fontSize: 18, fontWeight: 700, letterSpacing: "-0.5px", borderBottom: "1px solid #e8e6e1", marginBottom: 16 }}>CMS 管理</div>
        <nav style={{ flex: 1, padding: "0 12px" }}>
          <SL href="/admin">控制台</SL>
          <SL href="/admin/posts">文章管理</SL>
          <SL href="/admin/projects">项目管理</SL>
          <SL href="/admin/settings">站点设置</SL>
        </nav>
        <div style={{ padding: "0 12px", borderTop: "1px solid #e8e6e1", paddingTop: 16 }}>
          <SL href="/">← 返回前台</SL>
          <LogoutButton />
        </div>
      </aside>
      <main style={{ flex: 1, padding: "32px 40px", maxWidth: 960 }}>{children}</main>
    </div>
  );
}

function SL({ href, children }: { href: string; children: ReactNode }) {
  return <Link href={href} style={{ display: "block", padding: "8px 12px", borderRadius: 8, color: "#555", textDecoration: "none", fontSize: 14, marginBottom: 2 }}>{children}</Link>;
}
