"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import type { Post } from "@/lib/db";

interface Props { post?: Post; }

export default function PostEditor({ post }: Props) {
  const router = useRouter();
  const isNew = !post;
  const [title, setTitle] = useState(post?.title || "");
  const [slug, setSlug] = useState(post?.slug || "");
  const [content, setContent] = useState(post?.content || "");
  const [excerpt, setExcerpt] = useState(post?.excerpt || "");
  const [tags, setTags] = useState(post?.tags?.join(", ") || "");
  const [published, setPublished] = useState(post?.published ?? true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");

  const autoSlug = useCallback((t: string) => {
    if (isNew || !slug) {
      setSlug(t.toLowerCase().replace(/\s+/g, "-").replace(/[^a-z0-9\u4e00-\u9fff-]/g, "").slice(0, 60));
    }
  }, [isNew, slug]);

  async function handleSave() {
    setSaving(true);
    setMessage("");
    const body = {
      title, slug, content, excerpt,
      tags: tags.split(",").map(t => t.trim()).filter(Boolean),
      published,
    };
    const url = isNew ? "/api/posts" : "/api/posts/" + post!.id;
    const method = isNew ? "POST" : "PUT";
    const res = await fetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (res.ok) {
      setMessage("保存成功！");
      if (isNew) router.push("/admin/posts");
    } else {
      setMessage("保存失败");
    }
    setSaving(false);
  }

  async function handleUploadImage(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    if (res.ok) {
      const { url } = await res.json();
      setContent(prev => prev + "\n![](" + url + ")\n");
    }
  }

  function insertMarkdown(syntax: string, placeholder: string) {
    setContent(prev => prev + syntax.replace("", placeholder));
  }

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
        <h1 style={{ fontSize: 28, fontWeight: 700, margin: 0 }}>{isNew ? "新建文章" : "编辑文章"}</h1>
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={handleSave} disabled={saving}
            style={{ padding: "10px 24px", borderRadius: 10, background: "#1c1c1e", color: "#fff", border: "none", fontSize: 14, fontWeight: 600, cursor: "pointer" }}>
            {saving ? "保存中..." : "保存"}
          </button>
        </div>
      </div>
      {message && <p style={{ padding: "10px 16px", borderRadius: 8, background: message.includes("失败") ? "#fee2e2" : "#dcfce7", fontSize: 13, marginBottom: 16 }}>{message}</p>}

      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <div>
          <label style={{ display: "block", fontSize: 13, fontWeight: 600, marginBottom: 6, color: "#555" }}>标题</label>
          <input value={title} onChange={e => { setTitle(e.target.value); autoSlug(e.target.value); }}
            style={{ width: "100%", padding: "10px 14px", borderRadius: 10, border: "1px solid #e0ded9", fontSize: 15, outline: "none", boxSizing: "border-box" }} placeholder="文章标题" />
        </div>
        <div>
          <label style={{ display: "block", fontSize: 13, fontWeight: 600, marginBottom: 6, color: "#555" }}>Slug (URL)</label>
          <input value={slug} onChange={e => setSlug(e.target.value)}
            style={{ width: "100%", padding: "10px 14px", borderRadius: 10, border: "1px solid #e0ded9", fontSize: 14, outline: "none", boxSizing: "border-box", fontFamily: "monospace" }} />
        </div>
        <div>
          <label style={{ display: "block", fontSize: 13, fontWeight: 600, marginBottom: 6, color: "#555" }}>摘要</label>
          <textarea value={excerpt} onChange={e => setExcerpt(e.target.value)} rows={2}
            style={{ width: "100%", padding: "10px 14px", borderRadius: 10, border: "1px solid #e0ded9", fontSize: 14, outline: "none", boxSizing: "border-box", resize: "vertical", fontFamily: "inherit" }} />
        </div>
        <div>
          <label style={{ display: "block", fontSize: 13, fontWeight: 600, marginBottom: 6, color: "#555" }}>标签 (逗号分隔)</label>
          <input value={tags} onChange={e => setTags(e.target.value)}
            style={{ width: "100%", padding: "10px 14px", borderRadius: 10, border: "1px solid #e0ded9", fontSize: 14, outline: "none", boxSizing: "border-box" }} placeholder="rPPG, 计算机视觉" />
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <input type="checkbox" id="published" checked={published} onChange={e => setPublished(e.target.checked)}
            style={{ width: 18, height: 18 }} />
          <label htmlFor="published" style={{ fontSize: 14 }}>发布</label>
        </div>

        <div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
            <label style={{ fontSize: 13, fontWeight: 600, color: "#555" }}>内容 (Markdown)</label>
            <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
              <ToolBtn onClick={() => insertMarkdown("## ", "标题")} label="H2" />
              <ToolBtn onClick={() => insertMarkdown("### ", "副标题")} label="H3" />
              <ToolBtn onClick={() => insertMarkdown("****", "粗体")} label="B" />
              <ToolBtn onClick={() => insertMarkdown("**", "斜体")} label="I" />
              <ToolBtn onClick={() => insertMarkdown("- ", "列表项")} label="List" />
              <ToolBtn onClick={() => insertMarkdown("[链接]()", "https://")} label="Link" />
              <label style={{ padding: "4px 10px", borderRadius: 6, background: "#fff", border: "1px solid #e0ded9", fontSize: 12, cursor: "pointer", color: "#555" }}>
                📷 <input type="file" accept="image/*" onChange={handleUploadImage} style={{ display: "none" }} />
              </label>
            </div>
          </div>
          <textarea value={content} onChange={e => setContent(e.target.value)} rows={20}
            style={{ width: "100%", padding: "14px", borderRadius: 10, border: "1px solid #e0ded9", fontSize: 14, outline: "none", boxSizing: "border-box", resize: "vertical", fontFamily: "monospace", lineHeight: 1.7 }} placeholder="用 Markdown 编写内容..." />
        </div>
      </div>
    </div>
  );
}

function ToolBtn({ onClick, label }: { onClick: () => void; label: string }) {
  return (
    <button onClick={onClick}
      style={{ padding: "4px 10px", borderRadius: 6, background: "#fff", border: "1px solid #e0ded9", fontSize: 12, cursor: "pointer", color: "#555", fontWeight: label === "B" ? 700 : 400, fontStyle: label === "I" ? "italic" : "normal" }}>
      {label}
    </button>
  );
}
