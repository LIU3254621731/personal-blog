import Link from "next/link";
import { getProjects, getPosts } from "@/lib/db";
import { formatDate } from "@/lib/utils";
import { getReadingTime } from "@/lib/reading-time";

export default function Home() {
  const projects = getProjects();
  const featuredProjects = projects.filter((p) => p.featured);
  const latestPosts = getPosts().filter(p => p.published).slice(0, 3);

  return (
    <div className="mx-auto max-w-3xl px-6">
      <section className="mb-28 animate-fade-up">
        <p className="font-mono text-[11px] tracking-[0.2em] text-text-tertiary uppercase mb-8">个人作品集</p>
        <h1 className="font-display text-4xl sm:text-5xl md:text-6xl mb-8 max-w-2xl font-semibold tracking-tight leading-[1.06]">
          在 AI、系统与设计<br />
          <span className="text-text-tertiary">的交汇处创造。</span>
        </h1>
        <div className="h-px w-12 bg-border-medium mb-8" />
        <p className="text-base sm:text-lg text-text-secondary max-w-xl leading-relaxed">
          这里记录了我的项目、思考和实验——涵盖计算机视觉、深度学习、嵌入式系统和全栈开发等领域。
        </p>
      </section>

      {/* Latest Posts */}
      {latestPosts.length > 0 && (
        <section className="mb-28">
          <div className="flex items-center justify-between mb-8">
            <h2 className="font-display text-2xl font-semibold">最新文章</h2>
            <Link href="/blog" className="text-sm text-text-tertiary hover:text-text-secondary transition-colors link-underline">查看全部 &rarr;</Link>
          </div>
          <div className="space-y-3 stagger">
            {latestPosts.map((post) => {
              const rt = getReadingTime(post.content);
              return (
                <Link key={post.id} href={"/blog/" + post.slug}>
                  <div className="glass rounded-2xl p-5 transition-all duration-400 glass-card-hover cursor-pointer group flex items-center justify-between gap-4">
                    <div className="min-w-0">
                      <h3 className="font-display text-base mb-1 leading-snug group-hover:text-accent-warm transition-colors truncate">{post.title}</h3>
                      <p className="text-xs text-text-tertiary line-clamp-1">{post.excerpt}</p>
                    </div>
                    <div className="flex items-center gap-3 text-[11px] text-text-tertiary shrink-0">
                      <span>{formatDate(post.createdAt)}</span>
                      <span className="w-1 h-1 rounded-full bg-border-medium" />
                      <span>{rt} 分钟</span>
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        </section>
      )}

      {/* Featured Projects */}
      <section className="mb-28">
        <div className="flex items-center justify-between mb-8">
          <h2 className="font-display text-2xl font-semibold">精选项目</h2>
          <Link href="/projects" className="text-sm text-text-tertiary hover:text-text-secondary transition-colors link-underline">查看全部 &rarr;</Link>
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          {featuredProjects.map((project) => (
            <Link key={project.id} href={"/projects#" + project.id}>
              <div className="glass rounded-2xl p-6 h-full transition-all duration-400 glass-card-hover cursor-pointer">
                <p className="font-mono text-[11px] tracking-[0.15em] text-text-tertiary uppercase mb-3">{project.category}</p>
                <h3 className="font-display text-xl mb-3 leading-tight">{project.title}</h3>
                <p className="text-sm text-text-secondary leading-relaxed line-clamp-3 mb-5">{project.description}</p>
                <div className="flex flex-wrap gap-1.5 mt-auto">
                  {project.tags.slice(0, 3).map((tag) => (<span key={tag} className="px-2.5 py-0.5 text-[11px] rounded-full bg-black/4 dark:bg-white/5 text-text-tertiary">{tag}</span>))}
                </div>
              </div>
            </Link>
          ))}
        </div>
      </section>

      <section className="mb-28">
        <div className="glass rounded-2xl p-8 sm:p-10">
          <h2 className="font-display text-2xl font-semibold mb-4">关于我</h2>
          <p className="text-text-secondary leading-relaxed max-w-xl mb-6">我是 Liu Wenlin。我的工作横跨 AI 研究、计算机视觉、嵌入式开发和全栈工程。我相信好的技术应该是隐形、可靠且优美的。</p>
          <Link href="/about" className="inline-flex items-center gap-2 text-sm text-text-tertiary hover:text-text-secondary transition-colors link-underline">了解更多<span className="text-xs">&rarr;</span></Link>
        </div>
      </section>
    </div>
  );
}
