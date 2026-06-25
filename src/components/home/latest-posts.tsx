import Link from "next/link";
import { getPosts } from "@/lib/db";
import { formatDate } from "@/lib/utils";
import { getReadingTime } from "@/lib/reading-time";

export function LatestPosts() {
  const posts = getPosts()
    .filter((p) => p.published)
    .slice(0, 5);
  if (posts.length === 0) return null;

  return (
    <section className="mb-10">
      <div className="flex items-center justify-between mb-6">
        <div>
          <p className="font-mono text-[11px] tracking-[0.15em] text-text-tertiary uppercase mb-1">
            Writing
          </p>
          <h2 className="font-display text-xl font-semibold text-text-primary">
            最新文章
          </h2>
        </div>
        <Link
          href="/blog"
          className="text-xs text-text-tertiary hover:text-accent transition-colors link-underline"
        >
          全部文章 →
        </Link>
      </div>

      <div className="space-y-2">
        {posts.map((post) => {
          const rt = getReadingTime(post.content);
          return (
            <Link key={post.id} href={`/blog/${post.slug}`}>
              <article className="group glass rounded-xl p-5 transition-all duration-300 glass-card-hover">
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <h3 className="font-display text-base font-medium mb-1.5 leading-snug text-text-primary group-hover:text-accent transition-colors">
                      {post.title}
                    </h3>
                    <p className="text-sm text-text-secondary line-clamp-2 leading-relaxed mb-3">
                      {post.excerpt}
                    </p>
                    <div className="flex items-center gap-2 flex-wrap">
                      {post.tags.map((tag) => (
                        <span
                          key={tag}
                          className="px-2 py-0.5 text-[10px] rounded-md bg-tag-bg text-text-tertiary font-medium"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="text-right shrink-0 pt-0.5">
                    <p className="text-xs text-text-tertiary whitespace-nowrap">
                      {formatDate(post.createdAt)}
                    </p>
                    <p className="text-[10px] text-text-tertiary mt-0.5">
                      {rt} min
                    </p>
                  </div>
                </div>
              </article>
            </Link>
          );
        })}
      </div>
    </section>
  );
}
