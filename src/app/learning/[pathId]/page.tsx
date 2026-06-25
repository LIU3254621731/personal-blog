"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowLeft, FileText } from "lucide-react";
import { formatDate } from "@/lib/utils";
import { InlineCreateButton } from "@/components/admin/InlineControls";

interface PathData {
  id: string; name: string; description: string; tags: string[];
}

interface Post {
  id: string; title: string; slug: string; excerpt: string; tags: string[];
  createdAt: string; published: boolean;
}

export default function LearningPathDetailPage({ params }: { params: Promise<{ pathId: string }> }) {
  // Using useState + useEffect pattern since we need dynamic data
  const [pathData, setPathData] = useState<PathData | null>(null);
  const [posts, setPosts] = useState<Post[]>([]);
  const [loading, setLoading] = useState(true);
  const [pathId, setPathId] = useState("");

  useEffect(() => {
    params.then(p => setPathId(p.pathId));
  }, [params]);

  useEffect(() => {
    if (!pathId) return;
    async function load() {
      // Load path data
      const pr = await fetch(`/api/learning-paths/${pathId}`);
      if (pr.ok) {
        const pd = await pr.json();
        setPathData(pd);
        
        // Load posts matching path tags
        const tags = pd.tags || [];
        const tagsQuery = tags.length > 0 ? `?tag=${encodeURIComponent(tags[0])}` : "";
        const postsRes = await fetch(`/api/posts${tagsQuery}`);
        if (postsRes.ok) {
          const allPosts = await postsRes.json();
          // Filter by ANY matching tag
          const filtered = allPosts.filter((p: Post) =>
            p.published && p.tags.some((t: string) => tags.includes(t))
          ).slice(0, 30);
          setPosts(filtered);
        }
      }
      setLoading(false);
    }
    load();
  }, [pathId]);

  if (loading) return <div className="text-center text-text-tertiary py-32">加载中...</div>;
  if (!pathData) return <div className="text-center text-text-tertiary py-32">路线不存在</div>;

  return (
    <div className="mx-auto max-w-3xl px-6 py-12">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <Link href="/learning" className="inline-flex items-center gap-1 text-sm text-text-tertiary hover:text-text-secondary transition-colors mb-6">
          <ArrowLeft size={14} /> 返回
        </Link>

        <header className="mb-8">
          <h1 className="font-display text-3xl font-semibold text-text-primary mb-2">{pathData.name}</h1>
          {pathData.description && <p className="text-text-secondary">{pathData.description}</p>}
          <div className="flex gap-1.5 mt-3">
            {pathData.tags.map(t => (
              <span key={t} className="px-2 py-0.5 text-[10px] rounded-full bg-tag-bg text-text-tertiary">{t}</span>
            ))}
          </div>
        </header>

        <div className="mb-6">
          <InlineCreateButton
            href={`/blog/new?tags=${encodeURIComponent(pathData.tags.join(","))}`}
            label="新建文章"
          />
        </div>

        {posts.length > 0 ? (
          <div className="space-y-2">
            {posts.map(post => (
              <Link key={post.id} href={`/blog/${post.slug}`}>
                <article className="glass rounded-xl p-4 transition-all duration-300 glass-card-hover group flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <h3 className="text-sm font-medium text-text-primary group-hover:text-accent transition-colors line-clamp-1">{post.title}</h3>
                    <p className="text-xs text-text-tertiary mt-1 line-clamp-1">{post.excerpt}</p>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <span className="text-[10px] text-text-tertiary">{formatDate(post.createdAt)}</span>
                    <FileText size={13} className="text-text-tertiary" />
                  </div>
                </article>
              </Link>
            ))}
          </div>
        ) : (
          <p className="text-text-tertiary text-center py-16">此学习路线暂无文章。创建新文章或为已有文章添加相应标签。</p>
        )}
      </motion.div>
    </div>
  );
}
