"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Save, ArrowLeft } from "lucide-react";
import Link from "next/link";
import type { GardenEntry } from "@/lib/db";

const CATEGORIES = [
  { v: "thought", l: "思考" },
  { v: "inspiration", l: "灵感" },
  { v: "observation", l: "观察" },
  { v: "startup", l: "创业" },
  { v: "product", l: "产品" },
];
const STAGES = [
  { v: "seedling", l: "🌱 种子" },
  { v: "bud", l: "🌿 萌芽" },
  { v: "evergreen", l: "🌳 常青" },
];

export function GardenEditorForm({ entry }: { entry?: GardenEntry }) {
  const router = useRouter();
  const isNew = !entry;
  const [title, setTitle] = useState(entry?.title || "");
  const [slug, setSlug] = useState(entry?.slug || "");
  const [content, setContent] = useState(entry?.content || "");
  const [excerpt, setExcerpt] = useState(entry?.excerpt || "");
  const [tags, setTags] = useState(entry?.tags?.join(", ") || "");
  const [category, setCategory] = useState(entry?.category || "thought");
  const [stage, setStage] = useState(entry?.stage || "seedling");
  const [published, setPublished] = useState(entry?.published ?? true);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<{ text: string; ok: boolean } | null>(null);

  function autoSlug(t: string) {
    if (isNew) setSlug(t.toLowerCase().replace(/\s+/g, "-").replace(/[^a-z0-9\u4e00-\u9fff-]/g, "").slice(0, 60));
  }

  async function handleSave() {
    setSaving(true); setMsg(null);
    const body = { title, slug, content, excerpt, tags: tags.split(",").map(t => t.trim()).filter(Boolean), category, stage, published };
    const url = isNew ? "/api/garden" : `/api/garden/${entry!.id}`;
    const method = isNew ? "POST" : "PUT";
    const csrf = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]*)/)?.[1] || "";
    const res = await fetch(url, { method, headers: { "Content-Type": "application/json", "x-csrf-token": csrf }, body: JSON.stringify(body) });
    if (res.ok) {
      setMsg({ text: "保存成功！", ok: true });
      router.push(isNew ? "/garden" : `/garden/${entry!.slug}`);
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
            <Link href="/garden" className="text-text-tertiary hover:text-text-primary transition-colors"><ArrowLeft size={18} /></Link>
            <h1 className="font-display text-2xl font-semibold text-text-primary">{isNew ? "新种子" : "编辑"}</h1>
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
        <div className="space-y-5">
          <input value={title} onChange={e => { setTitle(e.target.value); autoSlug(e.target.value); }} className={ic} placeholder="标题" />
          <input value={slug} onChange={e => setSlug(e.target.value)} className={`${ic} font-mono text-xs`} placeholder="slug" />
          <textarea value={excerpt} onChange={e => setExcerpt(e.target.value)} rows={2} className={ic} placeholder="摘要" />
          <div className="grid grid-cols-2 gap-4">
            <select value={category} onChange={e => setCategory(e.target.value)} className={ic}>
              {CATEGORIES.map(c => <option key={c.v} value={c.v}>{c.l}</option>)}
            </select>
            <select value={stage} onChange={e => setStage(e.target.value)} className={ic}>
              {STAGES.map(s => <option key={s.v} value={s.v}>{s.l}</option>)}
            </select>
          </div>
          <input value={tags} onChange={e => setTags(e.target.value)} className={ic} placeholder="标签，逗号分隔" />
          <label className="flex items-center gap-2 text-sm text-text-secondary cursor-pointer">
            <input type="checkbox" checked={published} onChange={e => setPublished(e.target.checked)} className="w-4 h-4 rounded accent-accent" />
            发布
          </label>
          <textarea value={content} onChange={e => setContent(e.target.value)} rows={18} className={`${ic} font-mono text-sm leading-relaxed resize-y`} placeholder="Markdown 内容..." />
        </div>
      </motion.div>
    </div>
  );
}
