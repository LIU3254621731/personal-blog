"use client";

import { useEffect, useState } from "react";
import type { Project } from "@/lib/db";

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [showForm, setShowForm] = useState(false);
  const [editId, setEditId] = useState<string | null>(null);
  const [form, setForm] = useState({ title: "", category: "", description: "", tags: "", featured: false });

  useEffect(() => { fetch("/api/projects").then(r => r.json()).then(setProjects); }, []);

  function resetForm() {
    setForm({ title: "", category: "", description: "", tags: "", featured: false });
    setEditId(null);
    setShowForm(false);
  }

  function startEdit(p: Project) {
    setForm({ title: p.title, category: p.category, description: p.description, tags: p.tags.join(", "), featured: p.featured });
    setEditId(p.id);
    setShowForm(true);
  }

  async function handleSave() {
    const body = {
      title: form.title, category: form.category, description: form.description,
      tags: form.tags.split(",").map(t => t.trim()).filter(Boolean), featured: form.featured,
    };
    if (editId) {
      const res = await fetch("/api/projects/" + editId, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (res.ok) {
        const updated = await res.json();
        setProjects(projects.map(p => p.id === editId ? updated : p));
      }
    } else {
      const res = await fetch("/api/projects", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (res.ok) {
        const created = await res.json();
        setProjects([...projects, created]);
      }
    }
    resetForm();
  }

  async function handleDelete(id: string) {
    if (!confirm("确定删除？")) return;
    await fetch("/api/projects/" + id, { method: "DELETE" });
    setProjects(projects.filter(p => p.id !== id));
  }

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 28 }}>
        <h1 style={{ fontSize: 28, fontWeight: 700, margin: 0 }}>项目管理</h1>
        <button onClick={() => setShowForm(true)}
          style={{ padding: "10px 20px", borderRadius: 10, background: "#1c1c1e", color: "#fff", border: "none", fontSize: 14, fontWeight: 600, cursor: "pointer" }}>
          + 新建
        </button>
      </div>

      {showForm && (
        <div style={{ background: "#fff", borderRadius: 14, padding: 24, marginBottom: 24, border: "1px solid #e8e6e1" }}>
          <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 16 }}>{editId ? "编辑项目" : "新建项目"}</h2>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <input value={form.title} onChange={e => setForm({ ...form, title: e.target.value })} placeholder="标题" style={inputStyle} />
            <input value={form.category} onChange={e => setForm({ ...form, category: e.target.value })} placeholder="分类" style={inputStyle} />
            <textarea value={form.description} onChange={e => setForm({ ...form, description: e.target.value })} placeholder="描述" rows={3} style={inputStyle} />
            <input value={form.tags} onChange={e => setForm({ ...form, tags: e.target.value })} placeholder="标签 (逗号分隔)" style={inputStyle} />
            <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 14 }}>
              <input type="checkbox" checked={form.featured} onChange={e => setForm({ ...form, featured: e.target.checked })} /> 精选
            </label>
            <div style={{ display: "flex", gap: 8 }}>
              <button onClick={handleSave} style={{ padding: "8px 20px", borderRadius: 8, background: "#1c1c1e", color: "#fff", border: "none", cursor: "pointer", fontSize: 14 }}>保存</button>
              <button onClick={resetForm} style={{ padding: "8px 20px", borderRadius: 8, background: "#f5f5f5", border: "1px solid #e0ded9", cursor: "pointer", fontSize: 14 }}>取消</button>
            </div>
          </div>
        </div>
      )}

      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {projects.map(p => (
          <div key={p.id} style={{ background: "#fff", borderRadius: 12, padding: "16px 20px", display: "flex", justifyContent: "space-between", alignItems: "center", border: "1px solid #e8e6e1" }}>
            <div>
              <p style={{ margin: "0 0 4px", fontWeight: 600, fontSize: 15 }}>{p.title} {p.featured && <span style={{ fontSize: 11, color: "#999", background: "#f0eee9", padding: "2px 8px", borderRadius: 10, marginLeft: 8 }}>精选</span>}</p>
              <p style={{ margin: 0, fontSize: 12, color: "#999" }}>{p.category}</p>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button onClick={() => startEdit(p)} style={{ padding: "6px 14px", borderRadius: 8, border: "1px solid #e0ded9", fontSize: 13, cursor: "pointer", background: "#fff" }}>编辑</button>
              <button onClick={() => handleDelete(p.id)} style={{ padding: "6px 14px", borderRadius: 8, border: "1px solid #fecaca", fontSize: 13, color: "#e53e3e", background: "#fff", cursor: "pointer" }}>删除</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  width: "100%", padding: "10px 14px", borderRadius: 10, border: "1px solid #e0ded9",
  fontSize: 14, outline: "none", boxSizing: "border-box", fontFamily: "inherit",
};
