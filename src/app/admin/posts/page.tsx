"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Plus, Pencil, Trash2, Eye } from "lucide-react";
import type { Post } from "@/lib/db";

export default function PostsPage() {
  const [posts, setPosts] = useState<Post[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/posts")
      .then((r) => r.json())
      .then(setPosts)
      .finally(() => setLoading(false));
  }, []);

  async function handleDelete(id: string) {
    if (!confirm("确定删除?")) return;
    await fetch("/api/posts/" + id, { method: "DELETE" });
    setPosts((prev) => prev.filter((p) => p.id !== id));
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div className="flex items-center justify-between mb-8">
        <h1 className="font-display text-2xl font-semibold text-text-primary">
          文章管理
        </h1>
        <Link
          href="/admin/posts/new"
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-accent text-white text-sm font-medium
            hover:bg-accent/90 active:scale-[0.98] transition-all duration-200"
        >
          <Plus size={17} />
          新建文章
        </Link>
      </div>

      {loading ? (
        <div className="text-center text-text-tertiary py-20">加载中...</div>
      ) : posts.length === 0 ? (
        <div className="text-center text-text-tertiary py-20">
          <p className="text-lg mb-2">暂无文章</p>
          <p className="text-sm">点击「新建文章」创建第一篇内容</p>
        </div>
      ) : (
        <div className="space-y-2">
          {posts.map((post, i) => (
            <motion.div
              key={post.id}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.03, duration: 0.3 }}
              className="flex items-center justify-between glass rounded-xl px-5 py-4
                hover:bg-bg-primary/80 transition-all duration-200"
            >
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <h3 className="font-medium text-text-primary truncate">
                    {post.title}
                  </h3>
                  <span
                    className={`shrink-0 text-[11px] px-2 py-0.5 rounded-full font-medium ${
                      post.published
                        ? "bg-green-100 text-green-700 dark:bg-green-500/15 dark:text-green-400"
                        : "bg-yellow-100 text-yellow-700 dark:bg-yellow-500/15 dark:text-yellow-400"
                    }`}
                  >
                    {post.published ? "已发布" : "草稿"}
                  </span>
                </div>
                <p className="text-xs text-text-tertiary">
                  /{post.slug} ·{" "}
                  {new Date(post.createdAt).toLocaleDateString("zh-CN")}
                </p>
              </div>
              <div className="flex items-center gap-1.5 ml-4">
                <Link
                  href={`/blog/${post.slug}`}
                  target="_blank"
                  className="p-2 rounded-lg text-text-tertiary hover:text-accent hover:bg-accent-light/50
                    transition-all duration-200"
                  title="前台查看"
                >
                  <Eye size={16} />
                </Link>
                <Link
                  href={"/admin/posts/" + post.id}
                  className="p-2 rounded-lg text-text-tertiary hover:text-text-primary hover:bg-black/3 dark:hover:bg-white/5
                    transition-all duration-200"
                  title="编辑"
                >
                  <Pencil size={16} />
                </Link>
                <button
                  onClick={() => handleDelete(post.id)}
                  className="p-2 rounded-lg text-text-tertiary hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-500/10
                    transition-all duration-200"
                  title="删除"
                >
                  <Trash2 size={16} />
                </button>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </motion.div>
  );
}
