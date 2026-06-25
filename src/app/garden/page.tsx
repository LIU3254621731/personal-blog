import { getGardenEntries } from "@/lib/db";
import { getMdxFiles } from "@/lib/mdx";
import { GardenCard } from "@/components/garden/garden-card";
import { GardenAdminBar, GardenAdminActions } from "@/components/admin/ProjectGardenAdminControls";

const STAGES = ["seedling", "bud", "evergreen"] as const;
const CATEGORIES = ["thought", "inspiration", "observation", "startup", "product"] as const;

const CATEGORY_NAMES: Record<string, string> = {
  thought: "思考",
  inspiration: "灵感",
  observation: "观察",
  startup: "创业",
  product: "产品分析",
};

export default function GardenPage() {
  const dbEntries = getGardenEntries();

  // Also get MDX-based garden entries
  const mdxEntries = getMdxFiles("garden");

  return (
    <div className="mx-auto max-w-5xl px-6">
      {/* Header */}
      <section className="mb-16 animate-fade-up">
        <p className="font-mono text-[11px] tracking-[0.15em] text-text-tertiary uppercase mb-4">
          Digital Garden
        </p>
        <h1 className="font-display text-4xl sm:text-5xl font-semibold tracking-tight mb-4">
          数字花园
        </h1>
        <p className="text-text-secondary max-w-xl leading-relaxed">
          思想的花园。灵感、想法、观察和思考在这里有机生长。不追求完美，拥抱持续迭代。
        </p>
      </section>

      {/* Garden stages legend */}
      <div className="flex flex-wrap gap-4 mb-12 text-sm">
        <div className="flex items-center gap-2">
          <span className="text-base">🌱</span>
          <span className="text-text-tertiary">Seedling — 刚种下的想法</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-base">🌿</span>
          <span className="text-text-tertiary">Bud — 发展中的想法</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-base">🌳</span>
          <span className="text-text-tertiary">Evergreen — 成熟的思考</span>
        </div>
      </div>

      {/* Category tabs */}
      <div className="flex flex-wrap gap-2 mb-10">
        {CATEGORIES.map((cat) => (
          <a
            key={cat}
            href={`#${cat}`}
            className="px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 bg-tag-bg text-text-secondary hover:text-accent hover:bg-accent-light dark:hover:bg-accent-light/20"
          >
            {CATEGORY_NAMES[cat] ?? cat}
          </a>
        ))}
      </div>

      {/* Admin bar */}
      <GardenAdminBar />

      {/* DB entries by category */}
      {CATEGORIES.map((cat) => {
        const items = dbEntries.filter((e) => e.category === cat);
        if (items.length === 0) return null;
        return (
          <div key={cat} id={cat} className="mb-16">
            <h2 className="font-display text-xl font-semibold mb-6">
              {CATEGORY_NAMES[cat] ?? cat}
            </h2>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {items.map((entry) => (
                <div key={entry.id}>
                  <GardenCard entry={entry} />
                  <GardenAdminActions entryId={entry.id} />
                </div>
              ))}
            </div>
          </div>
        );
      })}

      {/* MDX-based garden entries */}
      {mdxEntries.length > 0 && (
        <div className="mb-16">
          <h2 className="font-display text-xl font-semibold mb-6">MDX 花园</h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {mdxEntries.map((entry) => {
              const fm = entry.frontmatter;
              return (
                <a key={entry.slug} href={`/garden/${entry.slug}`}>
                  <div className="glass rounded-2xl p-5 h-full transition-all duration-300 glass-card-hover cursor-pointer flex flex-col">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-xs font-medium garden-seed">
                        🌱 Seedling
                      </span>
                      <span className="text-[10px] text-text-tertiary uppercase tracking-wider">
                        {(fm.category as string) ?? "thought"}
                      </span>
                    </div>
                    <h3 className="font-display text-lg font-medium mb-2 leading-snug">
                      {fm.title as string}
                    </h3>
                    <p className="text-sm text-text-secondary leading-relaxed line-clamp-3 mb-4 flex-1">
                      {entry.content.slice(0, 160)}
                    </p>
                    <div className="flex flex-wrap gap-1.5 mt-auto">
                      {((fm.tags as string[]) ?? []).slice(0, 4).map((tag: string) => (
                        <span
                          key={tag}
                          className="px-2 py-0.5 text-[10px] rounded-md bg-tag-bg text-text-tertiary font-medium"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                </a>
              );
            })}
          </div>
        </div>
      )}

      {dbEntries.length === 0 && mdxEntries.length === 0 && (
        <div className="glass rounded-2xl p-12 text-center mb-20">
          <span className="text-4xl mb-4 block">🌱</span>
          <p className="text-text-secondary mb-2">花园刚刚种下第一颗种子</p>
          <p className="text-xs text-text-tertiary">
            想法会在这里慢慢生长...
          </p>
        </div>
      )}
    </div>
  );
}
