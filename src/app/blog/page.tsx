import Link from "next/link";
import { getPosts } from "@/lib/db";
import { formatDate } from "@/lib/utils";
import { getReadingTime } from "@/lib/reading-time";

export default function BlogPage() {
  const posts = getPosts().filter(p => p.published);

  return (
    <div className="mx-auto max-w-3xl px-6">
      <header className="mb-16 animate-fade-up">
        <p className="font-mono text-[11px] tracking-[0.2em] text-text-tertiary uppercase mb-4">文章</p>
        <h1 className="font-display text-4xl sm:text-5xl font-semibold tracking-tight">关于代码、AI 与创造的思考。</h1>
      </header>
      <div className="space-y-4 stagger">
        {posts.map((post) => {
          const rt = getReadingTime(post.content);
          return (
            <Link key={post.id} href={"/blog/" + post.slug}>
              <div className="glass rounded-2xl p-6 transition-all duration-400 glass-card-hover cursor-pointer group">
                <div className="flex items-center gap-3 mb-2 text-[11px] text-text-tertiary">
                  <span>{formatDate(post.createdAt)}</span>
                  <span className="w-1 h-1 rounded-full bg-border-medium" />
                  <span>{rt} 分钟阅读</span>
                </div>
                <h3 className="font-display text-xl mb-2 leading-snug group-hover:text-accent-warm transition-colors duration-300">{post.title}</h3>
                <p className="text-sm text-text-secondary leading-relaxed">{post.excerpt}</p>
                <div className="flex flex-wrap gap-1.5 mt-4">
                  {post.tags.map((tag: string) => (<span key={tag} className="px-2.5 py-0.5 text-[11px] rounded-full bg-black/4 dark:bg-white/5 text-text-tertiary">{tag}</span>))}
                </div>
              </div>
            </Link>
          );
        })}
        {posts.length === 0 && <p className="text-text-tertiary text-center py-12">暂无文章，去后台创建第一篇吧。</p>}
      </div>
    </div>
  );
}
