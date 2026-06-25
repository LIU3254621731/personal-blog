import Link from "next/link";
import { getPosts } from "@/lib/db";

export default function TagsPage() {
  const posts = getPosts().filter((p) => p.published);

  // Build tag → posts map
  const tagMap = new Map<string, typeof posts>();
  for (const post of posts) {
    for (const tag of post.tags) {
      if (!tagMap.has(tag)) tagMap.set(tag, []);
      tagMap.get(tag)!.push(post);
    }
  }

  // Sort tags by post count
  const sortedTags = Array.from(tagMap.entries()).sort(
    (a, b) => b[1].length - a[1].length,
  );

  return (
    <div className="mx-auto max-w-3xl px-6">
      <header className="mb-12">
        <p className="font-mono text-[11px] tracking-[0.15em] text-text-tertiary uppercase mb-3">
          Tags
        </p>
        <h1 className="font-display text-3xl sm:text-4xl font-semibold tracking-tight text-text-primary">
          标签
        </h1>
      </header>

      {sortedTags.length === 0 ? (
        <p className="text-text-tertiary text-center py-12">暂无标签</p>
      ) : (
        <div className="space-y-8">
          {sortedTags.map(([tag, tagPosts]) => (
            <section key={tag}>
              <div className="flex items-baseline gap-3 mb-4 pb-2 border-b border-border-light">
                <Link
                  href={`/blog?tag=${encodeURIComponent(tag)}`}
                  className="text-lg font-display font-medium text-text-primary hover:text-accent transition-colors"
                >
                  {tag}
                </Link>
                <span className="text-xs text-text-tertiary">
                  {tagPosts.length} 篇
                </span>
              </div>
              <div className="space-y-2">
                {tagPosts.slice(0, 5).map((post) => (
                  <Link
                    key={post.id}
                    href={`/blog/${post.slug}`}
                    className="block text-sm text-text-secondary hover:text-accent transition-colors"
                  >
                    {post.title}
                  </Link>
                ))}
                {tagPosts.length > 5 && (
                  <Link
                    href={`/blog?tag=${encodeURIComponent(tag)}`}
                    className="text-xs text-text-tertiary hover:text-accent transition-colors"
                  >
                    查看全部 {tagPosts.length} 篇 →
                  </Link>
                )}
              </div>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}
