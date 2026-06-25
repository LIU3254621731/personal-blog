"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { SiteConfigAdminBar, SiteConfigAdminActions } from "@/components/admin/SiteConfigAdminControls";

interface Goal {
  key: string;
  title: string;
  description: string;
  progress: number;
  year: string;
}

const FIELDS = [
  { key: "title", label: "目标名称" },
  { key: "description", label: "描述", type: "textarea" as const },
  { key: "progress", label: "进度 (0-100)", type: "number" as const },
  { key: "year", label: "年份" },
];

export default function RoadmapPage() {
  const [goals, setGoals] = useState<Goal[]>([]);
  const [loading, setLoading] = useState(true);

  async function load() {
    const res = await fetch("/api/roadmap");
    const data = await res.json();
    const items: Goal[] = [];
    for (const [key, value] of Object.entries(data)) {
      if (!key.startsWith("roadmap_")) continue;
      try {
        const d = JSON.parse(value as string);
        items.push({ key, title: d.title || "", description: d.description || "", progress: d.progress || 0, year: d.year || "" });
      } catch {
        items.push({ key, title: value as string, description: "", progress: 0, year: "" });
      }
    }
    setGoals(items.sort((a, b) => b.year.localeCompare(a.year)));
    setLoading(false);
  }

  useEffect(() => { load(); }, []);

  if (loading) return <div className="text-center text-text-tertiary py-32">加载中...</div>;

  const years = [...new Set(goals.map(g => g.year))].sort().reverse();

  return (
    <div className="mx-auto max-w-3xl px-6 py-12">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <header className="mb-8">
          <p className="font-mono text-[11px] tracking-[0.15em] text-text-tertiary uppercase mb-3">Roadmap</p>
          <h1 className="font-display text-3xl font-semibold text-text-primary">成长路线图</h1>
          <p className="text-sm text-text-secondary mt-2">年度目标与阶段性里程碑</p>
        </header>

        <SiteConfigAdminBar prefix="roadmap" label="新目标" fields={FIELDS} onSave={load} />

        {years.map(year => {
          const yearGoals = goals.filter(g => g.year === year);
          if (yearGoals.length === 0) return null;
          return (
            <section key={year} className="mb-10">
              <h2 className="font-display text-xl font-semibold text-text-primary mb-4">{year}</h2>
              <div className="space-y-3">
                {yearGoals.map(g => (
                  <div key={g.key} className="glass rounded-2xl p-5 transition-all duration-300">
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <h3 className="font-medium text-text-primary">{g.title}</h3>
                        {g.description && <p className="text-xs text-text-tertiary mt-1">{g.description}</p>}
                      </div>
                      <span className="text-sm font-semibold text-accent ml-4 shrink-0">{g.progress}%</span>
                    </div>
                    <div className="h-2 rounded-full bg-tag-bg overflow-hidden">
                      <motion.div
                        className="h-full rounded-full bg-accent"
                        initial={{ width: 0 }}
                        animate={{ width: `${g.progress}%` }}
                        transition={{ duration: 0.8, delay: 0.1 }}
                      />
                    </div>
                    <div className="mt-3 flex justify-end">
                      <SiteConfigAdminActions
                        resourceKey={g.key}
                        prefix="roadmap"
                        fields={FIELDS}
                        currentData={g}
                        onDelete={load}
                        onSave={load}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </section>
          );
        })}

        {goals.length === 0 && (
          <p className="text-text-tertiary text-center py-16">暂无目标。登录后可以添加年度目标。</p>
        )}
      </motion.div>
    </div>
  );
}
