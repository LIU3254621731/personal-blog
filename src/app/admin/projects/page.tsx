"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Plus, Pencil, Trash2, Star, Eye } from "lucide-react";
import type { Project } from "@/lib/db";

const STATUS_OPTIONS = [
  { value: "planning", label: "规划中" },
  { value: "building", label: "开发中" },
  { value: "testing", label: "测试中" },
  { value: "released", label: "已发布" },
  { value: "archived", label: "已归档" },
] as const;

const STATUS_CLASSES: Record<string, string> = {
  planning:
    "bg-gray-100 text-gray-600 dark:bg-gray-500/15 dark:text-gray-400",
  building:
    "bg-indigo-100 text-indigo-600 dark:bg-indigo-500/15 dark:text-indigo-400",
  testing:
    "bg-yellow-100 text-yellow-600 dark:bg-yellow-500/15 dark:text-yellow-400",
  released:
    "bg-green-100 text-green-600 dark:bg-green-500/15 dark:text-green-400",
  archived:
    "bg-slate-100 text-slate-500 dark:bg-slate-500/15 dark:text-slate-400",
};

const inputClass =
  "w-full px-4 py-2.5 rounded-xl border border-border-medium bg-bg-primary/50 text-text-primary text-sm placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition-all duration-200";

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [showForm, setShowForm] = useState(false);
  const [editId, setEditId] = useState<string | null>(null);
  const [form, setForm] = useState({
    title: "",
    category: "",
    description: "",
    tags: "",
    featured: false,
    status: "building",
    githubUrl: "",
    demoUrl: "",
  });

  useEffect(() => {
    fetch("/api/projects")
      .then((r) => r.json())
      .then(setProjects);
  }, []);

  function resetForm() {
    setForm({
      title: "",
      category: "",
      description: "",
      tags: "",
      featured: false,
      status: "building",
      githubUrl: "",
      demoUrl: "",
    });
    setEditId(null);
    setShowForm(false);
  }

  function startEdit(p: Project) {
    setForm({
      title: p.title,
      category: p.category,
      description: p.description,
      tags: p.tags.join(", "),
      featured: p.featured,
      status: p.status,
      githubUrl: p.githubUrl,
      demoUrl: p.demoUrl,
    });
    setEditId(p.id);
    setShowForm(true);
  }

  async function handleSave() {
    const body = {
      title: form.title,
      category: form.category,
      description: form.description,
      tags: form.tags
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean),
      featured: form.featured,
      status: form.status,
      githubUrl: form.githubUrl,
      demoUrl: form.demoUrl,
    };

    if (editId) {
      const res = await fetch("/api/projects/" + editId, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (res.ok) {
        const updated = await res.json();
        setProjects((prev) => prev.map((p) => (p.id === editId ? updated : p)));
      }
    } else {
      const res = await fetch("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (res.ok) {
        const created = await res.json();
        setProjects((prev) => [...prev, created]);
      }
    }
    resetForm();
  }

  async function handleDelete(id: string) {
    if (!confirm("确定删除?")) return;
    await fetch("/api/projects/" + id, { method: "DELETE" });
    setProjects((prev) => prev.filter((p) => p.id !== id));
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div className="flex items-center justify-between mb-8">
        <h1 className="font-display text-2xl font-semibold text-text-primary">
          项目管理
        </h1>
        <button
          onClick={() => setShowForm(true)}
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-accent text-white text-sm font-medium
            hover:bg-accent/90 active:scale-[0.98] transition-all duration-200"
        >
          <Plus size={17} />
          新建项目
        </button>
      </div>

      {/* Form */}
      <AnimatePresence>
        {showForm && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden mb-6"
          >
            <div className="glass rounded-2xl p-6 space-y-4">
              <h2 className="font-display text-lg font-semibold text-text-primary">
                {editId ? "编辑项目" : "新建项目"}
              </h2>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <input
                  value={form.title}
                  onChange={(e) => setForm({ ...form, title: e.target.value })}
                  placeholder="项目标题"
                  className={inputClass}
                />
                <input
                  value={form.category}
                  onChange={(e) =>
                    setForm({ ...form, category: e.target.value })
                  }
                  placeholder="分类"
                  className={inputClass}
                />
              </div>

              <textarea
                value={form.description}
                onChange={(e) =>
                  setForm({ ...form, description: e.target.value })
                }
                placeholder="项目描述"
                rows={3}
                className={inputClass}
              />

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <input
                  value={form.tags}
                  onChange={(e) => setForm({ ...form, tags: e.target.value })}
                  placeholder="标签（逗号分隔）"
                  className={inputClass}
                />
                <select
                  value={form.status}
                  onChange={(e) =>
                    setForm({ ...form, status: e.target.value })
                  }
                  className={inputClass}
                >
                  {STATUS_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
                <label className="flex items-center gap-2 text-sm text-text-secondary cursor-pointer">
                  <input
                    type="checkbox"
                    checked={form.featured}
                    onChange={(e) =>
                      setForm({ ...form, featured: e.target.checked })
                    }
                    className="w-4 h-4 rounded accent-accent"
                  />
                  精选项目
                </label>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <input
                  value={form.githubUrl}
                  onChange={(e) =>
                    setForm({ ...form, githubUrl: e.target.value })
                  }
                  placeholder="GitHub URL"
                  className={inputClass}
                />
                <input
                  value={form.demoUrl}
                  onChange={(e) =>
                    setForm({ ...form, demoUrl: e.target.value })
                  }
                  placeholder="Demo URL"
                  className={inputClass}
                />
              </div>

              <div className="flex gap-3 pt-2">
                <button
                  onClick={handleSave}
                  className="px-5 py-2.5 rounded-xl bg-accent text-white text-sm font-medium
                    hover:bg-accent/90 active:scale-[0.98] transition-all duration-200"
                >
                  保存
                </button>
                <button
                  onClick={resetForm}
                  className="px-5 py-2.5 rounded-xl border border-border-medium text-text-secondary text-sm
                    hover:bg-black/3 dark:hover:bg-white/5 transition-all duration-200"
                >
                  取消
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* List */}
      <div className="space-y-2">
        {projects.map((p, i) => (
          <motion.div
            key={p.id}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.03, duration: 0.3 }}
            className="flex items-center justify-between glass rounded-xl px-5 py-4
              hover:bg-bg-primary/80 transition-all duration-200"
          >
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 mb-1">
                <h3 className="font-medium text-text-primary truncate">
                  {p.title}
                </h3>
                {p.featured && (
                  <Star
                    size={14}
                    className="text-warning shrink-0"
                    fill="currentColor"
                  />
                )}
                <span
                  className={`shrink-0 text-[11px] px-2 py-0.5 rounded-full font-medium ${STATUS_CLASSES[p.status] || ""}`}
                >
                  {STATUS_OPTIONS.find((o) => o.value === p.status)?.label ||
                    p.status}
                </span>
              </div>
              <p className="text-xs text-text-tertiary">{p.category}</p>
            </div>
            <div className="flex items-center gap-1.5 ml-4">
              {p.demoUrl && (
                <a
                  href={p.demoUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="p-2 rounded-lg text-text-tertiary hover:text-accent hover:bg-accent-light/50 transition-all duration-200"
                  title="查看 Demo"
                >
                  <Eye size={16} />
                </a>
              )}
              <button
                onClick={() => startEdit(p)}
                className="p-2 rounded-lg text-text-tertiary hover:text-text-primary hover:bg-black/3 dark:hover:bg-white/5 transition-all duration-200"
                title="编辑"
              >
                <Pencil size={16} />
              </button>
              <button
                onClick={() => handleDelete(p.id)}
                className="p-2 rounded-lg text-text-tertiary hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-500/10 transition-all duration-200"
                title="删除"
              >
                <Trash2 size={16} />
              </button>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
