import { getMdxSubdirs, getMdxFiles } from "@/lib/mdx";
import { getPosts } from "@/lib/db";
import { KnowledgeTree } from "@/components/learning/knowledge-tree";
import Link from "next/link";
import { BookOpen, Code, Cpu, FlaskConical, FileText } from "lucide-react";
import { LearningAdminBar } from "@/components/admin/LearningAdminControls";

const CATEGORY_ICONS: Record<string, React.ReactNode> = {
  ai: <Cpu size={22} />,
  ml: <FlaskConical size={22} />,
  python: <Code size={22} />,
  math: <BookOpen size={22} />,
};

const CATEGORY_NAMES: Record<string, string> = {
  ai: "AI / 人工智能",
  ml: "Machine Learning",
  python: "Python",
  math: "Math / 数学",
};

export default function LearningPage() {
  const subdirs = getMdxSubdirs("notes");
  const recentNotes = getMdxFiles("notes").slice(0, 6);

  // Pull blog posts tagged as learning content
  const learningTags = ["AI", "白皮书", "教程", "学习", "深度学习", "LLM", "RAG", "LangChain", "Agent", "架构", "系统设计"];
  const learningPosts = getPosts()
    .filter(p => p.published && p.tags.some(t => learningTags.includes(t)))
    .slice(0, 12);

  return (
    <div className="mx-auto max-w-5xl px-6">
      <section className="mb-12 animate-fade-up">
        <p className="font-mono text-[11px] tracking-[0.15em] text-text-tertiary uppercase mb-4">Learning</p>
        <h1 className="font-display text-4xl sm:text-5xl font-semibold tracking-tight mb-4">学习记录</h1>
        <p className="text-text-secondary max-w-xl leading-relaxed">
          系统化的知识地图，记录学习路径和深度理解。不是普通博客，而是可生长的知识体系。
        </p>
      </section>

      <LearningAdminBar />

      {/* Category cards */}
      <div className="mb-12">
        <h2 className="font-display text-xl font-semibold mb-6">学习路线</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {subdirs.map((dir) => (
            <Link key={dir} href={`/learning/${dir}`}>
              <div className="glass rounded-2xl p-5 transition-all duration-300 glass-card-hover cursor-pointer group">
                <span className="inline-flex items-center justify-center w-10 h-10 rounded-xl bg-accent-light dark:bg-accent-light/20 text-accent mb-3">
                  {CATEGORY_ICONS[dir] ?? <BookOpen size={22} />}
                </span>
                <h3 className="font-display text-lg font-medium mb-1 group-hover:text-accent transition-colors">
                  {CATEGORY_NAMES[dir] ?? dir}
                </h3>
              </div>
            </Link>
          ))}
        </div>
      </div>

      <div className="grid gap-10 lg:grid-cols-[1fr_280px]">
        {/* Main content */}
        <div>
          {/* Blog posts as learning material */}
          {learningPosts.length > 0 && (
            <section className="mb-12">
              <div className="flex items-center gap-2 mb-5">
                <FileText size={16} className="text-accent" />
                <h2 className="font-display text-lg font-semibold">白皮书 & 深度文章</h2>
              </div>
              <div className="space-y-2">
                {learningPosts.map((post) => (
                  <Link key={post.id} href={`/blog/${post.slug}`}>
                    <article className="glass rounded-xl p-4 transition-all duration-300 glass-card-hover group flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <h3 className="text-sm font-medium text-text-primary group-hover:text-accent transition-colors line-clamp-1">
                          {post.title}
                        </h3>
                        <p className="text-xs text-text-tertiary mt-1 line-clamp-1">{post.excerpt}</p>
                      </div>
                      <div className="flex gap-1 shrink-0">
                        {post.tags.slice(0, 3).map(t => (
                          <span key={t} className="text-[10px] px-1.5 py-0.5 rounded-full bg-tag-bg text-text-tertiary">{t}</span>
                        ))}
                      </div>
                    </article>
                  </Link>
                ))}
              </div>
              {learningPosts.length >= 12 && (
                <Link href="/blog?tag=白皮书" className="text-xs text-text-tertiary hover:text-accent transition-colors mt-3 inline-block">
                  查看全部 →
                </Link>
              )}
            </section>
          )}

          {/* MDX Notes */}
          {recentNotes.length > 0 && (
            <section>
              <h2 className="font-display text-lg font-semibold mb-5">最近笔记</h2>
              <div className="space-y-2">
                {recentNotes.map((note) => (
                  <Link key={note.slug} href={`/learning/${note.frontmatter.category || "ai"}/${note.slug}`}>
                    <div className="glass rounded-xl p-4 transition-all duration-300 glass-card-hover group flex items-center justify-between">
                      <div>
                        <h3 className="font-medium text-sm group-hover:text-accent transition-colors">
                          {note.frontmatter.title as string}
                        </h3>
                        <div className="flex items-center gap-2 mt-1">
                          <span className="text-[10px] text-text-tertiary uppercase tracking-wider">
                            {note.frontmatter.category as string}
                          </span>
                        </div>
                      </div>
                      <span className="text-[11px] text-text-tertiary shrink-0">
                        {note.frontmatter.date as string}
                      </span>
                    </div>
                  </Link>
                ))}
              </div>
            </section>
          )}
        </div>

        {/* Sidebar */}
        <aside className="hidden lg:block">
          <div className="sticky top-28">
            <KnowledgeTree />
          </div>
        </aside>
      </div>
    </div>
  );
}
