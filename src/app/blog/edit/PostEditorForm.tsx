"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Save, ArrowLeft } from "lucide-react";
import Link from "next/link";
import type { Post } from "@/lib/db";

interface Props {
  post?: Post;
}

export function PostEditorForm({ post }: Props) {
  const router = useRouter();
  const isNew = !post;
  const [title, setTitle] = useState(post?.title || "");
  const [slug, setSlug] = useState(post?.slug || "");
  const [content, setContent] = useState(post?.content || "");
  const [excerpt, setExcerpt] = useState(post?.excerpt || "");
  const [tags, setTags] = useState(post?.tags?.join(", ") || "");
  const [published, setPublished] = useState(post?.published ?? true);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<{ text: string; ok: boolean } | null>(null);

  function autoSlug(t: string) {
    if (isNew) {
      setSlug(t.toLowerCase().replace(/\s+/g, "-").replace(/[^a-z0-9\u4e00-\u9fff-]/g, "").slice(0, 60));
    }
  }

  async function handleSave() {
    setSaving(true);
    setMsg(null);
    const body = { title, slug, content, excerpt, tags: tags.split(",").map(t => t.trim()).filter(Boolean), published };
    const url = isNew ? "/api/posts" : `/api/posts/${post!.id}`;
    const method = isNew ? "POST" : "PUT";
    
    const csrf = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]*)/)?.[1] || "";
    const res = await fetch(url, {
      method,
      headers: { "Content-Type": "application/json", "x-csrf-token": csrf },
      body: JSON.stringify(body),
    });
    
    if (res.ok) {
      setMsg({ text: "保存成功！", ok: true });
      if (isNew) {
        const data = await res.json();
        if (data.slug) router.push(`/blog/${data.slug}`);
        else router.push("/blog");
      } else {
        router.refresh();
      }
    } else {
      const data = await res.json();
      setMsg({ text: data.error || "保存失败", ok: false });
    }
    setSaving(false);
  }

  const ic = "w-full px-4 py-2.5 rounded-xl border border-border-medium bg-bg-primary/50 text-text-primary text-sm placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition-all duration-200";

  return (
    <div className="mx-auto max-w-3xl px-6 py-8">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Link href="/blog" className="text-text-tertiary hover:text-text-primary transition-colors">
              <ArrowLeft size={18} />
            </Link>
            <h1 className="font-display text-2xl font-semibold text-text-primary">{isNew ? "新文章" : "编辑文章"}</h1>
          </div>
          <button onClick={handleSave} disabled={saving} className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-accent text-white text-sm font-medium hover:bg-accent/90 active:scale-[0.98] transition-all disabled:opacity-50">
            <Save size={16} />{saving ? "保存中..." : "保存"}
          </button>
        </div>

        {msg && (
          <motion.div initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} className={`mb-6 px-4 py-3 rounded-xl text-sm ${msg.ok ? "bg-green-50 text-green-700 dark:bg-green-500/10 dark:text-green-400" : "bg-red-50 text-red-600 dark:bg-red-500/10 dark:text-red-400"}`}>
            {msg.text}
          </motion.div>
        )}

        <div className="space-y-5">
          <input value={title} onChange={e => { setTitle(e.target.value); autoSlug(e.target.value); }} className={ic} placeholder="文章标题" />
          <input value={slug} onChange={e => setSlug(e.target.value)} className={`${ic} font-mono text-xs`} placeholder="url-slug" />
          <textarea value={excerpt} onChange={e => setExcerpt(e.target.value)} rows={2} className={ic} placeholder="摘要（可选）" />
          <input value={tags} onChange={e => setTags(e.target.value)} className={ic} placeholder="标签，逗号分隔" />
          <label className="flex items-center gap-2 text-sm text-text-secondary cursor-pointer">
            <input type="checkbox" checked={published} onChange={e => setPublished(e.target.checked)} className="w-4 h-4 rounded accent-accent" />
            发布
          </label>
          <textarea value={content} onChange={e => setContent(e.target.value)} rows={22} className={`${ic} font-mono text-sm leading-relaxed resize-y`} placeholder="Markdown 内容..." />
        </div>
      </motion.div>
    </div>
  );
}
