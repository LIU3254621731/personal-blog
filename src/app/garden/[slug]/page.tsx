import { notFound } from "next/navigation";
import Link from "next/link";
import type { Metadata } from "next";
import { getGardenEntryBySlug } from "@/lib/db";
import { getMdxBySlug, getAllMdxSlugs } from "@/lib/mdx";
import { ArrowLeft, Calendar } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const STAGE_ICONS: Record<string, { label: string; icon: string }> = {
  seedling: { label: "Seedling", icon: "🌱" },
  bud: { label: "Bud", icon: "🌿" },
  evergreen: { label: "Evergreen", icon: "🌳" },
};

const CATEGORY_NAMES: Record<string, string> = {
  thought: "思考",
  inspiration: "灵感",
  observation: "观察",
  startup: "创业",
  product: "产品分析",
};

interface Props {
  params: Promise<{ slug: string }>;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  // Try DB first, then MDX
  const entry = getGardenEntryBySlug(slug);
  if (entry) return { title: entry.title, description: entry.excerpt };

  const mdx = getMdxBySlug("garden", slug);
  if (mdx) return { title: mdx.frontmatter.title as string };

  return { title: "未找到" };
}

export default async function GardenEntryPage({ params }: Props) {
  const { slug } = await params;

  // Try DB entry first
  const dbEntry = getGardenEntryBySlug(slug);

  // Try MDX entry
  const mdxEntry = getMdxBySlug("garden", slug);

  if (!dbEntry && !mdxEntry) notFound();

  if (dbEntry) {
    const stage = STAGE_ICONS[dbEntry.stage] ?? STAGE_ICONS.seedling;

    return (
      <div className="mx-auto max-w-3xl px-6">
        <Link
          href="/garden"
          className="inline-flex items-center gap-2 text-sm text-text-tertiary hover:text-accent transition-colors mb-8"
        >
          <ArrowLeft size={16} />
          返回花园
        </Link>

        <section className="mb-10 animate-fade-up">
          <div className="flex items-center gap-3 mb-4">
            <span className="text-xs font-medium garden-seed">
              {stage.icon} {stage.label}
            </span>
            <span className="text-xs text-text-tertiary capitalize">
              {CATEGORY_NAMES[dbEntry.category] ?? dbEntry.category}
            </span>
          </div>

          <h1 className="font-display text-3xl sm:text-4xl font-semibold tracking-tight mb-4">
            {dbEntry.title}
          </h1>

          <div className="flex items-center gap-4 text-sm text-text-tertiary mb-6">
            <span className="flex items-center gap-1.5">
              <Calendar size={14} />
              {dbEntry.createdAt.split("T")[0]}
            </span>
          </div>

          <div className="flex flex-wrap gap-2 mb-8">
            {dbEntry.tags.map((tag) => (
              <span
                key={tag}
                className="px-2.5 py-0.5 text-[11px] rounded-full bg-tag-bg text-text-tertiary font-medium"
              >
                {tag}
              </span>
            ))}
          </div>
        </section>

        <div className="prose-custom mb-16">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {dbEntry.content}
          </ReactMarkdown>
        </div>
      </div>
    );
  }

  // MDX entry
  if (mdxEntry) {
    const fm = mdxEntry.frontmatter;
    const stage = STAGE_ICONS[(fm.stage as string) ?? "seedling"] ?? STAGE_ICONS.seedling;

    return (
      <div className="mx-auto max-w-3xl px-6">
        <Link
          href="/garden"
          className="inline-flex items-center gap-2 text-sm text-text-tertiary hover:text-accent transition-colors mb-8"
        >
          <ArrowLeft size={16} />
          返回花园
        </Link>

        <section className="mb-10 animate-fade-up">
          <div className="flex items-center gap-3 mb-4">
            <span className="text-xs font-medium garden-seed">
              {stage.icon} {stage.label}
            </span>
            <span className="text-xs text-text-tertiary capitalize">
              {CATEGORY_NAMES[(fm.category as string) ?? "thought"] ?? (fm.category as string)}
            </span>
          </div>

          <h1 className="font-display text-3xl sm:text-4xl font-semibold tracking-tight mb-4">
            {fm.title as string}
          </h1>

          <div className="flex items-center gap-4 text-sm text-text-tertiary mb-6">
            <span className="flex items-center gap-1.5">
              <Calendar size={14} />
              {fm.date as string}
            </span>
          </div>
        </section>

        <div className="prose-custom mb-16">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {mdxEntry.content}
          </ReactMarkdown>
        </div>
      </div>
    );
  }

  notFound();
}
