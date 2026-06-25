"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Save, ImagePlus, Bold, Italic, List, Link2, Heading2, Heading3 } from "lucide-react";
import type { Post } from "@/lib/db";
import { adminFetch } from "@/lib/admin-fetch";

interface Props {
  post?: Post;
}

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
  const [message, setMessage] = useState<{ text: string; ok: boolean } | null>(null);

  const autoSlug = useCallback(
    (t: string) => {
      if (isNew || !slug) {
        setSlug(
          t
            .toLowerCase()
            .replace(/\s+/g, "-")
            .replace(/[^a-z0-9\u4e00-\u9fff-]/g, "")
            .slice(0, 60),
        );
      }
    },
    [isNew, slug],
  );

  async function handleSave() {
    setSaving(true);
    setMessage(null);
    const body = {
      title,
      slug,
      content,
      excerpt,
      tags: tags
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean),
      published,
    };
    const url = isNew ? "/api/posts" : "/api/posts/" + post!.id;
    const method = isNew ? "POST" : "PUT";
    const res = await adminFetch(url, {
      method,
      body: JSON.stringify(body),
    });
    if (res.ok) {
      setMessage({ text: "保存成功！", ok: true });
      if (isNew) {
        const data = await res.json();
        router.push("/admin/posts/" + data.id);
      }
    } else {
      const data = await res.json();
      setMessage({ text: data.error || "保存失败", ok: false });
    }
    setSaving(false);
  }

  async function handleUploadImage(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    const csrf = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]*)/)?.[1] || "";
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData,
      headers: { "x-csrf-token": csrf },
    });
    if (res.ok) {
      const { url } = await res.json();
      setContent((prev) => prev + `\n![](${url})\n`);
    }
  }

  function insertMarkdown(syntax: string) {
    setContent((prev) => prev + syntax);
  }

  const inputClass =
    "w-full px-4 py-2.5 rounded-xl border border-border-medium bg-bg-primary/50 text-text-primary text-sm placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition-all duration-200";
  const labelClass = "block text-xs font-semibold text-text-secondary mb-1.5 uppercase tracking-wide";

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div className="flex items-center justify-between mb-8">
        <h1 className="font-display text-2xl font-semibold text-text-primary">
          {isNew ? "新建文章" : "编辑文章"}
        </h1>
        <button
          onClick={handleSave}
          disabled={saving}
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-accent text-white text-sm font-medium
            hover:bg-accent/90 active:scale-[0.98] transition-all duration-200 disabled:opacity-50"
        >
          <Save size={17} />
          {saving ? "保存中..." : "保存"}
        </button>
      </div>

      {message && (
        <motion.div
          initial={{ opacity: 0, y: -6 }}
          animate={{ opacity: 1, y: 0 }}
          className={`mb-6 px-4 py-3 rounded-xl text-sm ${
            message.ok
              ? "bg-green-50 text-green-700 dark:bg-green-500/10 dark:text-green-400"
              : "bg-red-50 text-red-600 dark:bg-red-500/10 dark:text-red-400"
          }`}
        >
          {message.text}
        </motion.div>
      )}

      <div className="space-y-5">
        {/* Title */}
        <div>
          <label className={labelClass}>标题</label>
          <input
            value={title}
            onChange={(e) => {
              setTitle(e.target.value);
              autoSlug(e.target.value);
            }}
            className={inputClass}
            placeholder="文章标题"
          />
        </div>

        {/* Slug */}
        <div>
          <label className={labelClass}>Slug (URL)</label>
          <input
            value={slug}
            onChange={(e) => setSlug(e.target.value)}
            className={`${inputClass} font-mono text-sm`}
            placeholder="article-slug"
          />
        </div>

        {/* Excerpt */}
        <div>
          <label className={labelClass}>摘要</label>
          <textarea
            value={excerpt}
            onChange={(e) => setExcerpt(e.target.value)}
            rows={2}
            className={inputClass}
            placeholder="简短描述..."
          />
        </div>

        {/* Tags & Published */}
        <div className="grid grid-cols-[1fr_auto] gap-4 items-end">
          <div>
            <label className={labelClass}>标签 (逗号分隔)</label>
            <input
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              className={inputClass}
              placeholder="rPPG, 计算机视觉"
            />
          </div>
          <label className="flex items-center gap-2.5 px-4 py-2.5 rounded-xl border border-border-medium cursor-pointer hover:bg-bg-secondary transition-colors select-none">
            <input
              type="checkbox"
              checked={published}
              onChange={(e) => setPublished(e.target.checked)}
              className="w-4 h-4 rounded accent-accent"
            />
            <span className="text-sm text-text-secondary font-medium">发布</span>
          </label>
        </div>

        {/* Content */}
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <label className={labelClass + " mb-0"}>内容 (Markdown)</label>
            <div className="flex items-center gap-1">
              <ToolBtn onClick={() => insertMarkdown("## **")} icon={Heading2} label="H2" />
              <ToolBtn onClick={() => insertMarkdown("### **")} icon={Heading3} label="H3" />
              <ToolBtn onClick={() => insertMarkdown("****")} icon={Bold} label="Bold" />
              <ToolBtn onClick={() => insertMarkdown("**")} icon={Italic} label="Italic" />
              <ToolBtn onClick={() => insertMarkdown("- **")} icon={List} label="List" />
              <ToolBtn onClick={() => insertMarkdown("[**](https://)")} icon={Link2} label="Link" />
              <label className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-border-medium text-xs text-text-secondary cursor-pointer hover:bg-accent-light/50 hover:text-accent transition-all duration-200">
                <ImagePlus size={14} />
                图片
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleUploadImage}
                  className="hidden"
                />
              </label>
            </div>
          </div>
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            rows={20}
            className={`${inputClass} font-mono text-sm leading-relaxed resize-y`}
            placeholder="用 Markdown 编写内容..."
          />
        </div>
      </div>
    </motion.div>
  );
}

function ToolBtn({
  onClick,
  icon: Icon,
  label,
}: {
  onClick: () => void;
  icon: React.ComponentType<{ size?: number }>;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-border-medium text-xs text-text-secondary
        hover:bg-accent-light/50 hover:text-accent transition-all duration-200"
      title={label}
    >
      <Icon size={13} />
    </button>
  );
}
