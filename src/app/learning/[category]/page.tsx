import { notFound } from "next/navigation";
import Link from "next/link";
import type { Metadata } from "next";
import { getMdxFiles, getMdxSubdirs } from "@/lib/mdx";
import { ArrowLeft, BookOpen, Clock } from "lucide-react";

const CATEGORY_NAMES: Record<string, string> = {
  ai: "AI / 人工智能",
  ml: "Machine Learning",
  python: "Python",
  math: "Math / 数学",
};

interface Props {
  params: Promise<{ category: string }>;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { category } = await params;
  return {
    title: CATEGORY_NAMES[category] ?? category,
    description: `学习笔记 — ${CATEGORY_NAMES[category] ?? category}`,
  };
}

export function generateStaticParams() {
  return getMdxSubdirs("notes").map((dir) => ({ category: dir }));
}

export default async function LearningCategoryPage({ params }: Props) {
  const { category } = await params;
  const subdirs = getMdxSubdirs("notes");
  if (!subdirs.includes(category)) notFound();

  const files = getMdxFiles(`notes/${category}`);

  return (
    <div className="mx-auto max-w-5xl px-6">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm text-text-tertiary mb-8">
        <Link href="/learning" className="hover:text-accent transition-colors">
          学习记录
        </Link>
        <span>/</span>
        <span className="text-text-secondary">
          {CATEGORY_NAMES[category] ?? category}
        </span>
      </div>

      <h1 className="font-display text-3xl font-semibold tracking-tight mb-2">
        {CATEGORY_NAMES[category] ?? category}
      </h1>
      <p className="text-text-tertiary mb-10">
        {files.length} 篇笔记
      </p>

      {files.length > 0 ? (
        <div className="space-y-3">
          {files.map((note) => (
            <Link
              key={note.slug}
              href={`/learning/${category}/${note.slug}`}
            >
              <div className="glass rounded-2xl p-5 transition-all duration-300 glass-card-hover cursor-pointer group flex items-center justify-between gap-4">
                <div className="min-w-0">
                  <h3 className="font-display text-base mb-1 group-hover:text-accent transition-colors truncate">
                    {note.frontmatter.title as string}
                  </h3>
                  <div className="flex items-center gap-2 text-[10px] text-text-tertiary">
                    <span className="uppercase tracking-wider">
                      {category}
                    </span>
                    {!!(note.frontmatter.difficulty as string) && (
                      <span className="bg-tag-bg px-1.5 py-0.5 rounded-full">
                        {note.frontmatter.difficulty as string}
                      </span>
                    )}
                    {!!(note.frontmatter.concept as string) && (
                      <span className="truncate max-w-[200px]">
                        {note.frontmatter.concept as string}
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-3 text-[11px] text-text-tertiary shrink-0">
                  <span>{note.frontmatter.date as string}</span>
                </div>
              </div>
            </Link>
          ))}
        </div>
      ) : (
        <div className="glass rounded-2xl p-12 text-center">
          <BookOpen size={32} className="mx-auto text-text-tertiary mb-4" />
          <p className="text-text-secondary mb-2">此分类暂无笔记</p>
          <p className="text-xs text-text-tertiary">
            在 <code className="bg-tag-bg px-1 py-0.5 rounded">content/notes/{category}/</code> 添加 Markdown 文件
          </p>
        </div>
      )}
    </div>
  );
}
