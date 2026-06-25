import { notFound } from "next/navigation";
import Link from "next/link";
import type { Metadata } from "next";
import { getGardenEntryBySlug } from "@/lib/db";
import { formatDate } from "@/lib/utils";
import { ArrowLeft, Calendar } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";

const STAGE_ICONS: Record<string, { icon: string; label: string }> = {
  seedling: { icon: "🌱", label: "种子" },
  bud: { icon: "🌿", label: "萌芽" },
  evergreen: { icon: "🌳", label: "常青" },
};

interface Props { params: Promise<{ slug: string }> }

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const entry = getGardenEntryBySlug(slug);
  if (entry) return { title: entry.title, description: entry.excerpt };
  return { title: "未找到" };
}

export default async function GardenDetailPage({ params }: Props) {
  const { slug } = await params;
  const dbEntry = getGardenEntryBySlug(slug);
  if (!dbEntry) notFound();

  const stage = STAGE_ICONS[dbEntry.stage] ?? STAGE_ICONS.seedling;
  return (
    <div className="mx-auto max-w-3xl px-6 py-10">
      <Link href="/garden" className="inline-flex items-center gap-1 text-sm text-text-tertiary hover:text-text-secondary transition-colors mb-8">
        <ArrowLeft size={14} /> 全部种子
      </Link>
      <article>
        <header className="mb-8">
          <div className="flex items-center gap-3 mb-3 text-xs text-text-tertiary">
            <span>{stage.icon} {stage.label}</span>
            <span className="w-1 h-1 rounded-full bg-border-medium" />
            <span className="inline-flex items-center gap-1"><Calendar size={12} />{formatDate(dbEntry.createdAt)}</span>
          </div>
          <h1 className="font-display text-3xl font-semibold tracking-tight mb-3 text-text-primary">{dbEntry.title}</h1>
          {dbEntry.tags.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {dbEntry.tags.map(tag => (
                <span key={tag} className="px-2.5 py-0.5 text-[11px] rounded-full bg-tag-bg text-text-tertiary">{tag}</span>
              ))}
            </div>
          )}
        </header>
        <div className="prose-custom">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeHighlight]}
            components={{
              img: ({ src, alt }) => (
                <img src={src} alt={alt || ""} className="rounded-xl my-8 w-full" loading="lazy" />
              ),
              a: ({ href, children }) => (
                <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
              ),
              pre: ({ children }) => (
                <pre className="bg-black/3 dark:bg-white/5 rounded-xl p-5 my-6 overflow-x-auto text-sm leading-relaxed">{children}</pre>
              ),
              code: ({ children, className }) => {
                const isInline = !className;
                if (isInline) return <code className="bg-black/5 dark:bg-white/8 px-1.5 py-0.5 rounded text-[0.9em]">{children}</code>;
                return <code className={className}>{children}</code>;
              },
              blockquote: ({ children }) => (
                <blockquote className="border-l-2 border-border-medium pl-5 my-6 italic text-text-tertiary">{children}</blockquote>
              ),
              table: ({ children }) => (
                <div className="overflow-x-auto my-6"><table className="w-full text-sm border-collapse">{children}</table></div>
              ),
              th: ({ children }) => (
                <th className="border border-border-light px-3 py-2 text-left font-semibold bg-black/2 dark:bg-white/3">{children}</th>
              ),
              td: ({ children }) => (
                <td className="border border-border-light px-3 py-2">{children}</td>
              ),
            }}
          >
            {dbEntry.content}
          </ReactMarkdown>
        </div>
      </article>
    </div>
  );
}
