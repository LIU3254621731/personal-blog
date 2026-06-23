"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

export default function AdminDashboard() {
  const [stats, setStats] = useState({ posts: 0, projects: 0 });

  useEffect(() => {
    fetch("/api/posts").then(r => r.json()).then(d => setStats(s => ({ ...s, posts: d.length })));
    fetch("/api/projects").then(r => r.json()).then(d => setStats(s => ({ ...s, projects: d.length })));
  }, []);

  return (
    <div>
      <h1 style={{ fontSize: 28, fontWeight: 700, marginBottom: 32 }}>控制台</h1>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 16, marginBottom: 40 }}>
        <StatCard label="文章数" value={stats.posts} href="/admin/posts" />
        <StatCard label="项目数" value={stats.projects} href="/admin/projects" />
      </div>
      <div style={{ display: "flex", gap: 12 }}>
        <Link href="/admin/posts/new" style={{ padding: "10px 20px", borderRadius: 10, background: "#1c1c1e", color: "#fff", textDecoration: "none", fontSize: 14, fontWeight: 600 }}>
          + 新建文章
        </Link>
        <Link href="/admin/projects" style={{ padding: "10px 20px", borderRadius: 10, background: "#fff", color: "#1c1c1e", textDecoration: "none", fontSize: 14, fontWeight: 500, border: "1px solid #e0ded9" }}>
          管理项目
        </Link>
      </div>
    </div>
  );
}

function StatCard({ label, value, href }: { label: string; value: number; href: string }) {
  return (
    <Link href={href} style={{ background: "#fff", borderRadius: 14, padding: "24px 28px", textDecoration: "none", border: "1px solid #e8e6e1" }}>
      <p style={{ color: "#999", fontSize: 13, margin: "0 0 8px" }}>{label}</p>
      <p style={{ fontSize: 36, fontWeight: 700, margin: 0, color: "#1c1c1e" }}>{value}</p>
    </Link>
  );
}
