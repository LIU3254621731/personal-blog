"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { FileText, FolderKanban, Plus, ArrowRight } from "lucide-react";

export default function AdminDashboard() {
  const [stats, setStats] = useState({ posts: 0, projects: 0 });

  useEffect(() => {
    fetch("/api/posts")
      .then((r) => r.json())
      .then((d) => setStats((s) => ({ ...s, posts: Array.isArray(d) ? d.length : 0 })));
    fetch("/api/projects")
      .then((r) => r.json())
      .then((d) => setStats((s) => ({ ...s, projects: Array.isArray(d) ? d.length : 0 })));
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.22, 0.61, 0.36, 1] }}
    >
      <h1 className="font-display text-2xl font-semibold text-text-primary mb-8">
        控制台
      </h1>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-4 mb-8">
        <StatCard
          icon={FileText}
          label="文章数"
          value={stats.posts}
          href="/admin/posts"
          delay={0}
        />
        <StatCard
          icon={FolderKanban}
          label="项目数"
          value={stats.projects}
          href="/admin/projects"
          delay={0.08}
        />
      </div>

      {/* Quick Actions */}
      <div className="flex flex-wrap gap-3">
        <Link
          href="/admin/posts/new"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-accent text-white text-sm font-medium
            hover:bg-accent/90 active:scale-[0.98] transition-all duration-200"
        >
          <Plus size={17} />
          新建文章
        </Link>
        <Link
          href="/admin/projects"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl border border-border-medium
            text-text-secondary hover:text-text-primary hover:bg-black/3 dark:hover:bg-white/5
            text-sm font-medium transition-all duration-200"
        >
          <ArrowRight size={17} />
          管理项目
        </Link>
        <Link
          href="/admin/settings"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl border border-border-medium
            text-text-secondary hover:text-text-primary hover:bg-black/3 dark:hover:bg-white/5
            text-sm font-medium transition-all duration-200"
        >
          <ArrowRight size={17} />
          站点设置
        </Link>
      </div>
    </motion.div>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  href,
  delay,
}: {
  icon: React.ComponentType<{ size?: number }>;
  label: string;
  value: number;
  href: string;
  delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.3 }}
    >
      <Link
        href={href}
        className="block glass rounded-xl p-6 glass-card-hover transition-all duration-300"
      >
        <div className="flex items-center gap-3 mb-3">
          <div className="w-9 h-9 rounded-lg bg-accent-light dark:bg-accent-light/20 flex items-center justify-center">
            <Icon size={18} />
          </div>
          <span className="text-sm text-text-tertiary">{label}</span>
        </div>
        <p className="text-3xl font-semibold text-text-primary font-display">
          {value}
        </p>
      </Link>
    </motion.div>
  );
}
