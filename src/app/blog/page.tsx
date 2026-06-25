import Link from "next/link";
import { getPosts } from "@/lib/db";
import { formatDate } from "@/lib/utils";
import { getReadingTime } from "@/lib/reading-time";
import { BlogAdminBar } from "@/components/admin/BlogAdminControls";
import { BlogAdminActions } from "@/components/admin/BlogAdminControls";

const PAGE_SIZE = 6;

interface Props {
  searchParams: Promise<{ page?: string; tag?: string }>;
}

export default async function BlogPage({ searchParams }: Props) {
  const params = await searchParams;
  const currentPage = Math.max(1, parseInt(params.page || "1", 10) || 1);
  const activeTag = params.tag || null;

  // Filter published posts
  let posts = getPosts().filter((p) => p.published);

  // Collect all unique tags
  const allTags = Array.from(new Set(posts.flatMap((p) => p.tags))).sort();

  // Filter by tag if selected
  if (activeTag) {
    posts = posts.filter((p) => p.tags.includes(activeTag));
  }

  // Paginate
  const totalPages = Math.ceil(posts.length / PAGE_SIZE);
  const pagedPosts = posts.slice(
    (currentPage - 1) * PAGE_SIZE,
    currentPage * PAGE_SIZE,
  );

  return (
    <div className="mx-auto max-w-3xl px-6">
      {/* Header */}
      <header className="mb-12">
        <p className="font-mono text-[11px] tracking-[0.15em] text-text-tertiary uppercase mb-3">
          Blog
        </p>
        <h1 className="font-display text-3xl sm:text-4xl font-semibold tracking-tight mb-4 text-text-primary">
          文章
        </h1>
        <p className="text-sm text-text-secondary leading-relaxed max-w-lg">
          关于代码、AI 与创造的思考。        </p>
      </header>

      {/* Tag filter */}
      {allTags.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-10">
          <Link
            href="/blog"
            className={`px-3 py-1 rounded-full text-xs font-medium transition-all duration-200 ${
              !activeTag
                ? "bg-accent text-white"
                : "bg-tag-bg text-text-tertiary hover:text-text-secondary hover:bg-black/5 dark:hover:bg-white/5"
            }`}
          >
            全部
          </Link>
          {allTags.map((tag) => (
            <Link
              key={tag}
              href={`/blog?tag=${encodeURIComponent(tag)}`}
              className={`px-3 py-1 rounded-full text-xs font-medium transition-all duration-200 ${
                activeTag === tag
                  ? "bg-accent text-white"
                  : "bg-tag-bg text-text-tertiary hover:text-text-secondary hover:bg-black/5 dark:hover:bg-white/5"
              }`}
            >
              {tag}
            </Link>
          ))}
        </div>
      )}

      <BlogAdminBar />

      {/* Posts */}
      {pagedPosts.length > 0 ? (
        <div className="space-y-4">
          {pagedPosts.map((post) => {
            const rt = getReadingTime(post.content);
            return (
              <Link key={post.id} href={`/blog/${post.slug}`}>
                <article className="glass rounded-2xl p-6 transition-all duration-300 glass-card-hover group">
                  <div className="flex items-center gap-3 mb-2 text-[11px] text-text-tertiary">
                    <span>{formatDate(post.createdAt)}</span>
                    <span className="w-1 h-1 rounded-full bg-border-medium" />
                    <span>{rt} 分钟阅读</span>
                  </div>
                  <h2 className="font-display text-xl font-medium mb-2 leading-snug text-text-primary group-hover:text-accent transition-colors">
                    {post.title}
                  </h2>
                  <p className="text-sm text-text-secondary leading-relaxed line-clamp-2 mb-3">
                    {post.excerpt}
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {post.tags.map((tag) => (
                      <span
                        key={tag}
                        className="px-2.5 py-0.5 text-[11px] rounded-full bg-tag-bg text-text-tertiary"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                  <BlogAdminActions postId={post.id} slug={post.slug} />
                </article>
              </Link>
            );
          })}
        </div>
      ) : (
        <p className="text-text-tertiary text-center py-16">
          {activeTag ? `没有找到标签为「{activeTag}」的文章` : "暂无文章"}
        </p>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <nav className="flex items-center justify-center gap-2 mt-12 pt-8 border-t border-border-light">
          {currentPage > 1 && (
            <Link
              href={`/blog?page=${currentPage - 1}${activeTag ? `&tag=${encodeURIComponent(activeTag)}` : ""}`}
              className="px-4 py-2 rounded-lg text-sm text-text-secondary hover:text-text-primary hover:bg-black/3 dark:hover:bg-white/5 transition-all"
            >
              ← 上一页            </Link>
          )}

          {Array.from({ length: totalPages }, (_, i) => i + 1).map((p) => (
            <Link
              key={p}
              href={`/blog?page=${p}${activeTag ? `&tag=${encodeURIComponent(activeTag)}` : ""}`}
              className={`w-9 h-9 rounded-lg text-sm font-medium flex items-center justify-center transition-all ${
                p === currentPage
                  ? "bg-accent text-white"
                  : "text-text-tertiary hover:text-text-primary hover:bg-black/3 dark:hover:bg-white/5"
              }`}
            >
              {p}
            </Link>
          ))}

          {currentPage < totalPages && (
            <Link
              href={`/blog?page=${currentPage + 1}${activeTag ? `&tag=${encodeURIComponent(activeTag)}` : ""}`}
              className="px-4 py-2 rounded-lg text-sm text-text-secondary hover:text-text-primary hover:bg-black/3 dark:hover:bg-white/5 transition-all"
            >
              下一页 →            </Link>
          )}
        </nav>
      )}
    </div>
  );
}
