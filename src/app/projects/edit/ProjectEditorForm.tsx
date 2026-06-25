"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Save, ArrowLeft } from "lucide-react";
import Link from "next/link";
import type { Project } from "@/lib/db";

const STATUSES = [
  { v: "idea", l: "想法" },
  { v: "planning", l: "规划中" },
  { v: "building", l: "开发中" },
  { v: "testing", l: "测试中" },
  { v: "released", l: "已发布" },
  { v: "archived", l: "已归档" },
];

export function ProjectEditorForm({ project }: { project?: Project }) {
  const router = useRouter();
  const isNew = !project;
  const [title, setTitle] = useState(project?.title || "");
  const [category, setCategory] = useState(project?.category || "");
  const [description, setDescription] = useState(project?.description || "");
  const [tags, setTags] = useState(project?.tags?.join(", ") || "");
  const [featured, setFeatured] = useState(project?.featured ?? false);
  const [status, setStatus] = useState(project?.status || "building");
  const [githubUrl, setGithubUrl] = useState(project?.githubUrl || "");
  const [demoUrl, setDemoUrl] = useState(project?.demoUrl || "");
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<{ text: string; ok: boolean } | null>(null);

  async function handleSave() {
    setSaving(true); setMsg(null);
    const body = {
      title, category, description,
      tags: tags.split(",").map((t) => t.trim()).filter(Boolean),
      featured, status, githubUrl, demoUrl,
    };
    const url = isNew ? "/api/projects" : `/api/projects/${project!.id}`;
    const method = isNew ? "POST" : "PUT";
    const csrf = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]*)/)?.[1] || "";
    const res = await fetch(url, {
      method, headers: { "Content-Type": "application/json", "x-csrf-token": csrf },
      body: JSON.stringify(body),
    });
    if (res.ok) {
      setMsg({ text: "保存成功！", ok: true });
      router.push(isNew ? "/projects" : `/projects/${project!.id}`);
    } else {
      const d = await res.json();
      setMsg({ text: d.error || "保存失败", ok: false });
    }
    setSaving(false);
  }

  const ic = "w-full px-4 py-2.5 rounded-xl border border-border-medium bg-bg-primary/50 text-text-primary text-sm placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition-all duration-200";

  return (
    <div className="mx-auto max-w-3xl px-6 py-8">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Link href="/projects" className="text-text-tertiary hover:text-text-primary transition-colors">
              <ArrowLeft size={18} />
            </Link>
            <h1 className="font-display text-2xl font-semibold text-text-primary">{isNew ? "新项目" : "编辑项目"}</h1>
          </div>
          <button onClick={handleSave} disabled={saving}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-accent text-white text-sm font-medium hover:bg-accent/90 active:scale-[0.98] transition-all disabled:opacity-50">
            <Save size={16} />{saving ? "保存中..." : "保存"}
          </button>
        </div>
        {msg && (
          <motion.div initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }}
            className={`mb-6 px-4 py-3 rounded-xl text-sm ${msg.ok ? "bg-green-50 text-green-700 dark:bg-green-500/10 dark:text-green-400" : "bg-red-50 text-red-600 dark:bg-red-500/10 dark:text-red-400"}`}>
            {msg.text}
          </motion.div>
        )}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
          <input value={title} onChange={(e) => setTitle(e.target.value)} className={ic} placeholder="项目名称" />
          <input value={category} onChange={(e) => setCategory(e.target.value)} className={ic} placeholder="分类" />
        </div>
        <div className="mt-5">
          <textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={3} className={ic} placeholder="项目描述" />
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-5 mt-5">
          <input value={tags} onChange={(e) => setTags(e.target.value)} className={ic} placeholder="标签，逗号分隔" />
          <select value={status} onChange={(e) => setStatus(e.target.value)} className={ic}>
            {STATUSES.map((s) => <option key={s.v} value={s.v}>{s.l}</option>)}
          </select>
          <label className="flex items-center gap-2 text-sm text-text-secondary cursor-pointer">
            <input type="checkbox" checked={featured} onChange={(e) => setFeatured(e.target.checked)} className="w-4 h-4 rounded accent-accent" />
            精选项目
          </label>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 mt-5">
          <input value={githubUrl} onChange={(e) => setGithubUrl(e.target.value)} className={ic} placeholder="GitHub URL" />
          <input value={demoUrl} onChange={(e) => setDemoUrl(e.target.value)} className={ic} placeholder="Demo URL" />
        </div>
      </motion.div>
    </div>
  );
}
