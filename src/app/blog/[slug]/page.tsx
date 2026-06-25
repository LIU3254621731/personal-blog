import Link from "next/link";
import { notFound } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { getPostBySlug, getPosts } from "@/lib/db";
import { formatDate } from "@/lib/utils";
import { getReadingTime } from "@/lib/reading-time";
import { ScrollProgress } from "@/components/blog/scroll-progress";
import { TableOfContents } from "@/components/blog/toc";
import { BackToTop } from "@/components/layout/back-to-top";

export default async function BlogPostPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const post = getPostBySlug(slug);
  if (!post) notFound();

  const readingTime = getReadingTime(post.content);
  const allPosts = getPosts().filter((p) => p.published);
  const idx = allPosts.findIndex((p) => p.slug === slug);
  const prev = idx > 0 ? allPosts[idx - 1] : null;
  const next = idx < allPosts.length - 1 ? allPosts[idx + 1] : null;

  // Related posts by tag overlap (exclude current)
  const related = allPosts
    .filter((p) => p.slug !== slug)
    .map((p) => {
      const overlap = p.tags.filter((t) => post.tags.includes(t)).length;
      return { ...p, overlap };
    })
    .filter((p) => p.overlap > 0)
    .sort((a, b) => b.overlap - a.overlap)
    .slice(0, 3);

  return (
    <>
      <ScrollProgress />
      <div className="mx-auto max-w-5xl px-6">
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_220px] gap-10">
          {/* Main content */}
          <article className="min-w-0">
            <Link
              href="/blog"
              className="inline-flex items-center gap-1 text-sm text-text-tertiary hover:text-text-secondary transition-colors mb-8"
            >
              ← 全部文章
            </Link>

            <header className="mb-12">
              <div className="flex items-center gap-4 mb-4 text-xs text-text-tertiary">
                <span>{formatDate(post.createdAt)}</span>
                <span className="w-1 h-1 rounded-full bg-border-medium" />
                <span>{readingTime} 分钟阅读</span>
              </div>
              <h1 className="font-display text-3xl sm:text-4xl font-semibold tracking-tight mb-4 text-text-primary">
                {post.title}
              </h1>
              <div className="flex flex-wrap gap-1.5">
                {post.tags.map((tag) => (
                  <Link
                    key={tag}
                    href={`/blog?tag=${encodeURIComponent(tag)}`}
                    className="px-2.5 py-0.5 text-[11px] rounded-full bg-tag-bg text-text-tertiary hover:text-accent hover:bg-accent-light/30 transition-colors"
                  >
                    {tag}
                  </Link>
                ))}
              </div>
            </header>

            <div className="prose-custom">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeHighlight]}
                components={{
                  img: ({ src, alt }) => (
                    <img
                      src={src}
                      alt={alt || ""}
                      className="rounded-xl my-8 w-full"
                      loading="lazy"
                    />
                  ),
                  a: ({ href, children }) => (
                    <a href={href} target="_blank" rel="noopener noreferrer">
                      {children}
                    </a>
                  ),
                  pre: ({ children }) => (
                    <pre className="bg-black/3 dark:bg-white/5 rounded-xl p-5 my-6 overflow-x-auto text-sm leading-relaxed">
                      {children}
                    </pre>
                  ),
                  code: ({ children, className }) => {
                    const isInline = !className;
                    if (isInline)
                      return (
                        <code className="bg-black/5 dark:bg-white/8 px-1.5 py-0.5 rounded text-[0.9em]">
                          {children}
                        </code>
                      );
                    return <code className={className}>{children}</code>;
                  },
                  blockquote: ({ children }) => (
                    <blockquote className="border-l-2 border-border-medium pl-5 my-6 italic text-text-tertiary">
                      {children}
                    </blockquote>
                  ),
                  table: ({ children }) => (
                    <div className="overflow-x-auto my-6">
                      <table className="w-full text-sm border-collapse">
                        {children}
                      </table>
                    </div>
                  ),
                  th: ({ children }) => (
                    <th className="border border-border-light px-3 py-2 text-left font-semibold bg-black/2 dark:bg-white/3">
                      {children}
                    </th>
                  ),
                  td: ({ children }) => (
                    <td className="border border-border-light px-3 py-2">
                      {children}
                    </td>
                  ),
                }}
              >
                {post.content}
              </ReactMarkdown>
            </div>

            {/* Prev / Next */}
            <nav className="mt-16 pt-8 border-t border-border-light grid grid-cols-2 gap-4">
              {prev ? (
                <Link
                  href={`/blog/${prev.slug}`}
                  className="glass rounded-xl p-4 hover:shadow-md transition-all group"
                >
                  <p className="text-[10px] tracking-wider text-text-tertiary uppercase mb-1">
                    ← 上一篇
                  </p>
                  <p className="text-sm font-medium text-text-primary group-hover:text-accent transition-colors line-clamp-1">
                    {prev.title}
                  </p>
                </Link>
              ) : (
                <div />
              )}
              {next ? (
                <Link
                  href={`/blog/${next.slug}`}
                  className="glass rounded-xl p-4 hover:shadow-md transition-all group text-right"
                >
                  <p className="text-[10px] tracking-wider text-text-tertiary uppercase mb-1">
                    下一篇 →
                  </p>
                  <p className="text-sm font-medium text-text-primary group-hover:text-accent transition-colors line-clamp-1">
                    {next.title}
                  </p>
                </Link>
              ) : (
                <div />
              )}
            </nav>

            {/* Related posts */}
            {related.length > 0 && (
              <section className="mt-16 pt-8 border-t border-border-light">
                <h2 className="font-display text-lg font-semibold text-text-primary mb-4">
                  相关文章
                </h2>
                <div className="space-y-2">
                  {related.map((p) => (
                    <Link key={p.id} href={`/blog/${p.slug}`}>
                      <div className="glass rounded-xl p-4 hover:shadow-md transition-all group">
                        <h3 className="text-sm font-medium text-text-primary group-hover:text-accent transition-colors line-clamp-1">
                          {p.title}
                        </h3>
                        <p className="text-xs text-text-tertiary mt-1 line-clamp-1">
                          {p.excerpt}
                        </p>
                      </div>
                    </Link>
                  ))}
                </div>
              </section>
            )}
          </article>

          {/* Sidebar: TOC */}
          <aside className="hidden lg:block">
            <div className="sticky top-28">
              <TableOfContents />
            </div>
          </aside>
        </div>
      </div>
      <BackToTop />
    </>
  );
}

export function generateStaticParams() {
  return getPosts().map((post) => ({ slug: post.slug }));
}
