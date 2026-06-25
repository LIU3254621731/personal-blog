"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { ExternalLink, Star } from "lucide-react";
import { SiteConfigAdminBar, SiteConfigAdminActions } from "@/components/admin/SiteConfigAdminControls";

interface Resource {
  key: string;
  title: string;
  url: string;
  category: string;
  tags: string[];
  rating: number;
  description: string;
}

const CATEGORY_NAMES: Record<string, string> = {
  books: "📚 书籍", courses: "🎓 课程", papers: "📄 论文",
  tools: "🔧 工具", websites: "🌐 网站", other: "📌 其他",
};

const FIELDS = [
  { key: "title", label: "资源名称" },
  { key: "url", label: "链接" },
  { key: "category", label: "分类 (books/courses/papers/tools/websites/other)" },
  { key: "tags", label: "标签（逗号分隔）" },
  { key: "rating", label: "评分 (1-5)", type: "number" as const },
  { key: "description", label: "描述", type: "textarea" as const },
];

function Rating({ value }: { value: number }) {
  return <div className="flex gap-0.5">{Array.from({ length: 5 }, (_, i) => (
    <Star key={i} size={11} className={i < value ? "text-warning fill-warning" : "text-border-medium"} />
  ))}</div>;
}

export default function ResourcesPage() {
  const [resources, setResources] = useState<Resource[]>([]);
  const [activeCat, setActiveCat] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  async function load() {
    const res = await fetch("/api/resources");
    const data = await res.json();
    const items: Resource[] = [];
    for (const [key, value] of Object.entries(data)) {
      if (!key.startsWith("resource_")) continue;
      try {
        const d = JSON.parse(value as string);
        items.push({ key, title: d.title || "", url: d.url || "", category: d.category || "other", tags: d.tags || [], rating: d.rating || 0, description: d.description || "" });
      } catch { /* skip */ }
    }
    setResources(items);
    setLoading(false);
  }

  useEffect(() => { load(); }, []);

  const categories = [...new Set(resources.map(r => r.category))].sort();
  const filtered = activeCat ? resources.filter(r => r.category === activeCat) : resources;

  if (loading) return <div className="text-center text-text-tertiary py-32">加载中...</div>;

  return (
    <div className="mx-auto max-w-3xl px-6 py-12">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <header className="mb-8">
          <p className="font-mono text-[11px] tracking-[0.15em] text-text-tertiary uppercase mb-3">Resources</p>
          <h1 className="font-display text-3xl font-semibold text-text-primary">资源库</h1>
          <p className="text-sm text-text-secondary mt-2">收集优质学习资源和工具</p>
        </header>

        <SiteConfigAdminBar prefix="resources" label="新资源" fields={FIELDS} onSave={load} />

        {/* Category filter */}
        {categories.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-8">
            <button onClick={() => setActiveCat(null)}
              className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${!activeCat ? "bg-accent text-white" : "bg-tag-bg text-text-tertiary"}`}>
              全部
            </button>
            {categories.map(c => (
              <button key={c} onClick={() => setActiveCat(c)}
                className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${activeCat === c ? "bg-accent text-white" : "bg-tag-bg text-text-tertiary"}`}>
                {CATEGORY_NAMES[c] || c}
              </button>
            ))}
          </div>
        )}

        <div className="space-y-3">
          {filtered.map(r => (
            <div key={r.key} className="glass rounded-xl p-4 transition-all duration-300 glass-card-hover">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <a href={r.url || "#"} target="_blank" rel="noreferrer"
                      className="font-medium text-sm text-text-primary hover:text-accent transition-colors inline-flex items-center gap-1">
                      {r.title} {r.url && <ExternalLink size={11} className="text-text-tertiary" />}
                    </a>
                    <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-tag-bg text-text-tertiary">{CATEGORY_NAMES[r.category] || r.category}</span>
                  </div>
                  {r.description && <p className="text-xs text-text-secondary line-clamp-2 mb-2">{r.description}</p>}
                  <div className="flex items-center gap-3">
                    <Rating value={r.rating} />
                    {r.tags.length > 0 && (
                      <div className="flex gap-1">{r.tags.map(t => (
                        <span key={t} className="text-[10px] text-text-tertiary bg-tag-bg px-1.5 py-0.5 rounded">{t}</span>
                      ))}</div>
                    )}
                  </div>
                </div>
              </div>
              <div className="mt-2 flex justify-end">
                <SiteConfigAdminActions resourceKey={r.key} prefix="resources" fields={FIELDS} currentData={r} onDelete={load} onSave={load} />
              </div>
            </div>
          ))}
        </div>

        {resources.length === 0 && (
          <p className="text-text-tertiary text-center py-16">暂无资源。登录后可以添加学习资源。</p>
        )}
      </motion.div>
    </div>
  );
}
