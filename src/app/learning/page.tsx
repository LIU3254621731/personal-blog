import { getMdxSubdirs, getMdxFiles } from "@/lib/mdx";
import { KnowledgeTree } from "@/components/learning/knowledge-tree";
import Link from "next/link";
import { BookOpen, Code, Cpu, FlaskConical } from "lucide-react";
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

  return (
    <div className="mx-auto max-w-5xl px-6">
      {/* Header */}
      <section className="mb-16 animate-fade-up">
        <p className="font-mono text-[11px] tracking-[0.15em] text-text-tertiary uppercase mb-4">
          Learning
        </p>
        <h1 className="font-display text-4xl sm:text-5xl font-semibold tracking-tight mb-4">
          学习记录
        </h1>
        <p className="text-text-secondary max-w-xl leading-relaxed">
          系统化的知识地图，记录学习路径和深度理解。不是普通博客，而是可生长的知识体系。
        </p>
      </section>

      <LearningAdminBar />

      <div className="grid gap-8 lg:grid-cols-[1fr_280px] mb-20">
        {/* Main: category cards */}
        <div>
          <h2 className="font-display text-xl font-semibold mb-6">学习路线</h2>
          <div className="grid gap-4 sm:grid-cols-2">
            {subdirs.map((dir) => {
              const files = getMdxFiles(`notes/${dir}`);
              return (
                <Link key={dir} href={`/learning/${dir}`}>
                  <div className="glass rounded-2xl p-5 transition-all duration-300 glass-card-hover cursor-pointer group">
                    <span className="inline-flex items-center justify-center w-10 h-10 rounded-xl bg-accent-light dark:bg-accent-light/20 text-accent mb-3">
                      {CATEGORY_ICONS[dir] ?? <BookOpen size={22} />}
                    </span>
                    <h3 className="font-display text-lg font-medium mb-1 group-hover:text-accent transition-colors">
                      {CATEGORY_NAMES[dir] ?? dir}
                    </h3>
                    <p className="text-xs text-text-tertiary">
                      {files.length} 篇笔记
                    </p>
                  </div>
                </Link>
              );
            })}
            {subdirs.length === 0 && (
              <p className="text-text-tertiary text-sm col-span-2 py-8 text-center">
                学习笔记即将上线。在 <code className="text-xs bg-tag-bg px-1.5 py-0.5 rounded">content/notes/</code> 目录添加 Markdown 文件即可。
              </p>
            )}
          </div>

          {/* Recent notes */}
          {recentNotes.length > 0 && (
            <div className="mt-12">
              <h2 className="font-display text-xl font-semibold mb-6">最近笔记</h2>
              <div className="space-y-3">
                {recentNotes.map((note) => (
                  <Link
                    key={note.slug}
                    href={`/learning/${note.frontmatter.category || "ai"}/${note.slug}`}
                  >
                    <div className="glass rounded-2xl p-4 transition-all duration-300 glass-card-hover cursor-pointer group flex items-center justify-between">
                      <div>
                        <h3 className="font-medium text-sm group-hover:text-accent transition-colors">
                          {note.frontmatter.title as string}
                        </h3>
                        <div className="flex items-center gap-2 mt-1">
                          <span className="text-[10px] text-text-tertiary uppercase tracking-wider">
                            {note.frontmatter.category as string}
                          </span>
                          {!!(note.frontmatter.difficulty as string) && (
                            <span className="text-[10px] text-text-tertiary bg-tag-bg px-1.5 py-0.5 rounded-full">
                              {note.frontmatter.difficulty as string}
                            </span>
                          )}
                        </div>
                      </div>
                      <span className="text-[11px] text-text-tertiary shrink-0">
                        {note.frontmatter.date as string}
                      </span>
                    </div>
                  </Link>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Sidebar: knowledge tree */}
        <aside className="hidden lg:block">
          <div className="sticky top-28">
            <KnowledgeTree />
          </div>
        </aside>
      </div>
    </div>
  );
}
