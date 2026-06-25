import { notFound } from "next/navigation";
import Link from "next/link";
import type { Metadata } from "next";
import { getMdxBySlug, getMdxFiles, getMdxSubdirs } from "@/lib/mdx";
import { ArrowLeft, ArrowRight, Calendar, Tag, Lightbulb, BookOpen } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const CATEGORY_NAMES: Record<string, string> = {
  ai: "AI / 人工智能",
  ml: "Machine Learning",
  python: "Python",
  math: "Math / 数学",
};

interface Props {
  params: Promise<{ category: string; slug: string }>;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { category, slug } = await params;
  const note = getMdxBySlug(`notes/${category}`, slug);
  if (!note) return { title: "笔记未找到" };
  return {
    title: note.frontmatter.title as string,
    description: note.frontmatter.concept as string ?? note.content.slice(0, 160),
  };
}

export function generateStaticParams() {
  const params: { category: string; slug: string }[] = [];
  for (const dir of getMdxSubdirs("notes")) {
    for (const file of getMdxFiles(`notes/${dir}`, { includeDrafts: true })) {
      params.push({ category: dir, slug: file.slug });
    }
  }
  return params;
}

export default async function LearningArticlePage({ params }: Props) {
  const { category, slug } = await params;
  const note = getMdxBySlug(`notes/${category}`, slug);
  if (!note) notFound();

  const fm = note.frontmatter;
  const title = (fm.title as string) ?? "";
  const date = (fm.date as string) ?? "";
  const difficulty = (fm.difficulty as string) ?? "";
  const concept = (fm.concept as string) ?? "";
  const tags = (fm.tags as string[]) ?? [];
  const prerequisites = (fm.prerequisites as string[]) ?? [];

  return (
    <div className="mx-auto max-w-3xl px-6">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm text-text-tertiary mb-8">
        <Link href="/learning" className="hover:text-accent transition-colors">
          学习记录
        </Link>
        <span>/</span>
        <Link href={`/learning/${category}`} className="hover:text-accent transition-colors">
          {CATEGORY_NAMES[category] ?? category}
        </Link>
        <span>/</span>
        <span className="text-text-secondary truncate max-w-[200px]">
          {title}
        </span>
      </div>

      {/* Article header */}
      <section className="mb-10 animate-fade-up">
        <h1 className="font-display text-3xl sm:text-4xl font-semibold tracking-tight mb-4">
          {title}
        </h1>

        {/* Metadata */}
        <div className="flex flex-wrap items-center gap-4 text-sm text-text-tertiary mb-4">
          {date && (
            <span className="flex items-center gap-1.5">
              <Calendar size={14} />
              {date}
            </span>
          )}
          {difficulty && (
            <span className="bg-tag-bg px-2 py-0.5 rounded-full text-xs">
              {difficulty.charAt(0).toUpperCase() + difficulty.slice(1)}
            </span>
          )}
        </div>

        {/* Concept */}
        {concept && (
          <div className="glass rounded-2xl p-4 mb-4 flex items-start gap-3">
            <Lightbulb size={18} className="text-accent shrink-0 mt-0.5" />
            <div>
              <p className="text-xs text-text-tertiary mb-0.5">核心概念</p>
              <p className="text-sm text-text-primary font-medium">
                {concept}
              </p>
            </div>
          </div>
        )}

        {/* Prerequisites */}
        {prerequisites.length > 0 && (
          <div className="flex items-center gap-2 text-xs text-text-tertiary mb-4">
            <BookOpen size={13} />
            <span>前置知识：</span>
            {prerequisites.map((p, i) => (
              <span key={i} className="bg-tag-bg px-2 py-0.5 rounded-full">
                {p}
              </span>
            ))}
          </div>
        )}

        {/* Tags */}
        {tags.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {tags.map((tag) => (
              <span
                key={tag}
                className="px-2.5 py-0.5 text-[11px] rounded-full bg-tag-bg text-text-tertiary font-medium"
              >
                {tag}
              </span>
            ))}
          </div>
        )}
      </section>

      {/* Article content */}
      <div className="prose-custom mb-16">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {note.content}
        </ReactMarkdown>
      </div>

      {/* Back link */}
      <div className="border-t border-border-light pt-8 pb-20">
        <Link
          href={`/learning/${category}`}
          className="inline-flex items-center gap-2 text-sm text-text-tertiary hover:text-accent transition-colors"
        >
          <ArrowLeft size={16} />
          返回 {CATEGORY_NAMES[category] ?? category}
        </Link>
      </div>
    </div>
  );
}
