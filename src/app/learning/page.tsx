"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { BookOpen, Cpu, Code, FlaskConical, Brain, GitBranch, Database, Globe, Wrench } from "lucide-react";
import { useRouter } from "next/navigation";
import { AdminFormModal } from "@/components/admin/AdminFormModal";

interface LearningPath {
  id: string;
  name: string;
  description: string;
  icon: string;
  tags: string[];
  order: number;
  _postCount?: number;
}

const ICON_MAP: Record<string, React.ReactNode> = {
  cpu: <Cpu size={20} />, code: <Code size={20} />, flask: <FlaskConical size={20} />,
  brain: <Brain size={20} />, book: <BookOpen size={20} />, git: <GitBranch size={20} />,
  database: <Database size={20} />, globe: <Globe size={20} />, wrench: <Wrench size={20} />,
};

const ICON_OPTIONS = Object.keys(ICON_MAP);

const PATH_FIELDS = [
  { key: "name", label: "路线名称" },
  { key: "description", label: "描述", type: "textarea" as const },
  { key: "icon", label: `图标 (${ICON_OPTIONS.join("/")})` },
  { key: "tags", label: "关联标签（逗号分隔）" },
  { key: "order", label: "排序", type: "number" as const },
];

function toStr(v: any): string {
  if (typeof v === "string") return v;
  if (Array.isArray(v)) return v.join(", ");
  if (v != null) return String(v);
  return "";
}

export default function LearningPage() {
  const [paths, setPaths] = useState<LearningPath[]>([]);
  const [postCounts, setPostCounts] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);
  const [authed, setAuthed] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [editPath, setEditPath] = useState<LearningPath | null>(null);
  const router = useRouter();

  async function load() {
    const [pathsRes, postsRes, authRes] = await Promise.all([
      fetch("/api/learning-paths"),
      fetch("/api/posts"),
      fetch("/api/auth/check"),
    ]);
    const pathsData = await pathsRes.json();
    const postsData = await postsRes.json();
    const authData = await authRes.json();
    setAuthed(authData.authenticated);

    // Count posts per path by tag matching
    const counts: Record<string, number> = {};
    for (const p of pathsData) {
      const tags = Array.isArray(p.tags) ? p.tags : (p.tags ? String(p.tags).split(",").map((t: string) => t.trim()) : []);
      counts[p.id] = postsData.filter((post: any) =>
        post.published && post.tags.some((t: string) => tags.includes(t))
      ).length;
    }
    setPostCounts(counts);
    setPaths(pathsData);
    setLoading(false);
  }

  useEffect(() => { load(); }, []);

  async function handleDelete(id: string) {
    if (!confirm("确定删除此学习路线？")) return;
    const csrf = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]*)/)?.[1] || "";
    await fetch(`/api/learning-paths/${id}`, { method: "DELETE", headers: { "x-csrf-token": csrf } });
    load();
  }

  async function handleSave(data: Record<string, any>) {
    const csrf = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]*)/)?.[1] || "";
    const tags = typeof data.tags === "string" ? data.tags.split(",").map((t: string) => t.trim()).filter(Boolean) : data.tags || [];
    const body = { ...data, tags };

    if (editPath) {
      await fetch(`/api/learning-paths/${editPath.id}`, {
        method: "PUT", headers: { "Content-Type": "application/json", "x-csrf-token": csrf },
        body: JSON.stringify(body),
      });
    } else {
      await fetch("/api/learning-paths", {
        method: "POST", headers: { "Content-Type": "application/json", "x-csrf-token": csrf },
        body: JSON.stringify(body),
      });
    }
    setModalOpen(false);
    setEditPath(null);
    load();
  }

  function openEdit(p: LearningPath) {
    setEditPath(p);
    setModalOpen(true);
  }

  function openCreate() {
    setEditPath(null);
    setModalOpen(true);
  }

  if (loading) return <div className="text-center text-text-tertiary py-32">加载中...</div>;

  return (
    <div className="mx-auto max-w-5xl px-6 py-12">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <header className="mb-10">
          <p className="font-mono text-[11px] tracking-[0.15em] text-text-tertiary uppercase mb-3">Learning</p>
          <div className="flex items-center justify-between">
            <div>
              <h1 className="font-display text-3xl font-semibold text-text-primary">学习路线</h1>
              <p className="text-sm text-text-secondary mt-2">系统化的知识体系，标签自动聚合文章</p>
            </div>
            {authed && (
              <button onClick={openCreate}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-accent text-white text-sm font-medium hover:bg-accent/90 transition-all">
                新建路线
              </button>
            )}
          </div>
        </header>

        {paths.length === 0 ? (
          <div className="text-center py-20 text-text-tertiary">
            <BookOpen size={40} className="mx-auto mb-4 opacity-30" />
            <p>{authed ? "暂无学习路线，点击「新建路线」创建第一条" : "暂无学习路线"}</p>
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {paths.map((p) => {
              const tags = Array.isArray(p.tags) ? p.tags : (p.tags ? String(p.tags).split(",").map((t: string) => t.trim()) : []);
              const count = postCounts[p.id] || 0;
              return (
                <div key={p.id} className="glass rounded-2xl p-5 transition-all duration-300 glass-card-hover relative group">
                  <Link href={`/learning/${p.id}`}>
                    <div className="flex items-center gap-3 mb-3">
                      <span className="inline-flex items-center justify-center w-10 h-10 rounded-xl bg-accent-light dark:bg-accent-light/20 text-accent">
                        {ICON_MAP[p.icon] || <BookOpen size={20} />}
                      </span>
                      <div>
                        <h3 className="font-display text-lg font-medium text-text-primary group-hover:text-accent transition-colors">
                          {p.name}
                        </h3>
                        <span className="text-xs text-text-tertiary">{count} 篇文章</span>
                      </div>
                    </div>
                    {p.description && <p className="text-xs text-text-secondary line-clamp-2">{p.description}</p>}
                    {tags.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-3">
                        {tags.slice(0, 4).map(t => (
                          <span key={t} className="text-[10px] px-1.5 py-0.5 rounded-full bg-tag-bg text-text-tertiary">{t}</span>
                        ))}
                      </div>
                    )}
                  </Link>
                  {authed && (
                    <div className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity flex gap-1">
                      <button onClick={(e) => { e.preventDefault(); openEdit(p); }}
                        className="p-1.5 rounded-lg text-xs text-text-tertiary hover:text-accent hover:bg-accent-light/30 transition-all">编辑</button>
                      <button onClick={(e) => { e.preventDefault(); handleDelete(p.id); }}
                        className="p-1.5 rounded-lg text-xs text-text-tertiary hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-500/10 transition-all">删除</button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </motion.div>

      <AdminFormModal
        open={modalOpen}
        onClose={() => { setModalOpen(false); setEditPath(null); }}
        title={editPath ? "编辑学习路线" : "新建学习路线"}
        fields={PATH_FIELDS}
        initialData={editPath ? { ...editPath, tags: Array.isArray(editPath.tags) ? editPath.tags.join(", ") : editPath.tags } : undefined}
        onSave={handleSave}
      />
    </div>
  );
}
