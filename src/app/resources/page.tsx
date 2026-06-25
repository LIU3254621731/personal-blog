import { getSiteConfig } from "@/lib/db";
import { ResourceAdminControls } from "./ResourceAdminControls";

interface Resource {
  key: string;
  title: string;
  url: string;
  category: string;
  tags: string[];
  rating: number; // 1-5
  description: string;
}

const CATEGORY_NAMES: Record<string, string> = {
  books: "📚 书籍",
  courses: "🎓 课程",
  papers: "📄 论文",
  tools: "🔧 工具",
  websites: "🌐 网站",
  other: "📌 其他",
};

function parseResources(config: Record<string, string>): Resource[] {
  const resources: Resource[] = [];
  for (const [key, value] of Object.entries(config)) {
    if (!key.startsWith("resource_")) continue;
    try {
      const data = JSON.parse(value);
      resources.push({
        key,
        title: data.title ?? key.replace("resource_", ""),
        url: data.url ?? "",
        category: data.category ?? "other",
        tags: Array.isArray(data.tags) ? data.tags : [],
        rating: typeof data.rating === "number" ? Math.min(5, Math.max(0, data.rating)) : 0,
        description: data.description ?? "",
      });
    } catch {
      resources.push({
        key,
        title: value,
        url: "",
        category: "other",
        tags: [],
        rating: 0,
        description: "",
      });
    }
  }
  return resources;
}

function groupByCategory(resources: Resource[]): Map<string, Resource[]> {
  const grouped = new Map<string, Resource[]>();
  for (const r of resources) {
    const cat = r.category || "other";
    if (!grouped.has(cat)) grouped.set(cat, []);
    grouped.get(cat)!.push(r);
  }
  return grouped;
}

function renderStars(rating: number): string {
  if (rating === 0) return "";
  const full = Math.floor(rating);
  const half = rating - full >= 0.5 ? 1 : 0;
  const empty = 5 - full - half;
  return "★".repeat(full) + (half ? "½" : "") + "☆".repeat(empty);
}

export default function ResourcesPage() {
  const config = getSiteConfig();
  const resources = parseResources(config);
  const grouped = groupByCategory(resources);

  const categoryOrder = ["books", "courses", "papers", "tools", "websites", "other"];

  return (
    <div className="mx-auto max-w-5xl px-6">
      {/* Header */}
      <section className="mb-16 animate-fade-up">
        <p className="font-mono text-[11px] tracking-[0.15em] text-text-tertiary uppercase mb-4">
          Resources
        </p>
        <h1 className="font-display text-4xl sm:text-5xl font-semibold tracking-tight mb-4">
          资源
        </h1>
        <p className="text-text-secondary max-w-xl leading-relaxed">
          我筛选和收藏的优质资源，涵盖书籍、课程、论文和工具。
        </p>
      </section>

      {/* Admin bar */}
      <ResourceAdminControls resources={resources} />

      {/* Grouped by category */}
      {categoryOrder.map((cat) => {
        const items = grouped.get(cat);
        if (!items || items.length === 0) return null;
        return (
          <div key={cat} id={cat} className="mb-16">
            <h2 className="font-display text-xl font-semibold mb-6">
              {CATEGORY_NAMES[cat] ?? cat}
            </h2>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {items.map((r) => (
                <div key={r.key}>
                  <div className="glass rounded-2xl p-5 transition-all duration-300 glass-card-hover flex flex-col h-full">
                    {/* Category badge */}
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-[10px] text-text-tertiary uppercase tracking-wider">
                        {CATEGORY_NAMES[r.category] ?? r.category}
                      </span>
                      {r.rating > 0 && (
                        <span className="text-[11px] text-amber-500 font-medium">
                          {renderStars(r.rating)}
                        </span>
                      )}
                    </div>

                    {/* Title */}
                    <h3 className="font-display text-lg font-medium mb-2 leading-snug">
                      {r.url ? (
                        <a
                          href={r.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="hover:text-accent transition-colors"
                        >
                          {r.title}
                        </a>
                      ) : (
                        r.title
                      )}
                    </h3>

                    {/* Description */}
                    {r.description && (
                      <p className="text-sm text-text-secondary leading-relaxed line-clamp-2 mb-4 flex-1">
                        {r.description}
                      </p>
                    )}

                    {/* Tags */}
                    {r.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 mt-auto pt-3 border-t border-border-light">
                        {r.tags.slice(0, 5).map((tag) => (
                          <span
                            key={tag}
                            className="px-2 py-0.5 text-[10px] rounded-md bg-tag-bg text-text-tertiary font-medium"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  {/* Inline admin actions placeholder */}
                  <div id={`resource-actions-${r.key}`} />
                </div>
              ))}
            </div>
          </div>
        );
      })}

      {resources.length === 0 && (
        <div className="glass rounded-2xl p-12 text-center mb-20">
          <span className="text-4xl mb-4 block">📚</span>
          <p className="text-text-secondary mb-2">还没有收藏资源</p>
          <p className="text-xs text-text-tertiary">
            通过管理后台添加书籍、课程、工具等资源...
          </p>
        </div>
      )}
    </div>
  );
}
