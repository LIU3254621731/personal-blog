"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import type { Post } from "@/lib/db";

export default function PostsPage() {
  const [posts, setPosts] = useState<Post[]>([]);

  useEffect(() => {
    fetch("/api/posts").then(r => r.json()).then(setPosts);
  }, []);

  async function handleDelete(id: string) {
    if (!confirm("确定删除？")) return;
    await fetch("/api/posts/" + id, { method: "DELETE" });
    setPosts(posts.filter(p => p.id !== id));
  }

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 28 }}>
        <h1 style={{ fontSize: 28, fontWeight: 700, margin: 0 }}>文章管理</h1>
        <Link href="/admin/posts/new" style={{ padding: "10px 20px", borderRadius: 10, background: "#1c1c1e", color: "#fff", textDecoration: "none", fontSize: 14, fontWeight: 600 }}>
          + 新建
        </Link>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {posts.map(post => (
          <div key={post.id} style={{ background: "#fff", borderRadius: 12, padding: "16px 20px", display: "flex", justifyContent: "space-between", alignItems: "center", border: "1px solid #e8e6e1" }}>
            <div>
              <p style={{ margin: "0 0 4px", fontWeight: 600, fontSize: 15 }}>{post.title}</p>
              <p style={{ margin: 0, fontSize: 12, color: "#999" }}>
                {post.slug} · {post.published ? "已发布" : "草稿"} · {new Date(post.createdAt).toLocaleDateString("zh-CN")}
              </p>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <Link href={"/admin/posts/" + post.id} style={{ padding: "6px 14px", borderRadius: 8, border: "1px solid #e0ded9", fontSize: 13, color: "#555", textDecoration: "none" }}>
                编辑
              </Link>
              <button onClick={() => handleDelete(post.id)} style={{ padding: "6px 14px", borderRadius: 8, border: "1px solid #fecaca", fontSize: 13, color: "#e53e3e", background: "#fff", cursor: "pointer" }}>
                删除
              </button>
            </div>
          </div>
        ))}
        {posts.length === 0 && <p style={{ color: "#999", textAlign: "center", padding: 40 }}>暂无文章</p>}
      </div>
    </div>
  );
}
