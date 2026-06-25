import Link from "next/link";
import type { GardenEntry } from "@/lib/db";

const STAGE_ICONS: Record<string, { label: string; className: string }> = {
  seedling: { label: "🌱 Seedling", className: "garden-seed" },
  bud: { label: "🌿 Bud", className: "garden-bud" },
  evergreen: { label: "🌳 Evergreen", className: "garden-evergreen" },
};

const CATEGORY_NAMES: Record<string, string> = {
  thought: "思考",
  inspiration: "灵感",
  observation: "观察",
  startup: "创业",
  product: "产品分析",
};

interface Props {
  entry: GardenEntry;
}

export function GardenCard({ entry }: Props) {
  const stage = STAGE_ICONS[entry.stage] ?? STAGE_ICONS.seedling;

  return (
    <Link href={`/garden/${entry.slug}`}>
      <div className="glass rounded-2xl p-5 h-full transition-all duration-300 glass-card-hover cursor-pointer flex flex-col">
        {/* Stage indicator */}
        <div className="flex items-center justify-between mb-3">
          <span className={`text-xs font-medium ${stage.className}`}>
            {stage.label}
          </span>
          <span className="text-[10px] text-text-tertiary uppercase tracking-wider">
            {CATEGORY_NAMES[entry.category] ?? entry.category}
          </span>
        </div>

        {/* Title + excerpt */}
        <h3 className="font-display text-lg font-medium mb-2 leading-snug">
          {entry.title}
        </h3>
        <p className="text-sm text-text-secondary leading-relaxed line-clamp-3 mb-4 flex-1">
          {entry.excerpt}
        </p>

        {/* Tags */}
        <div className="flex flex-wrap gap-1.5 mt-auto">
          {entry.tags.slice(0, 4).map((tag) => (
            <span
              key={tag}
              className="px-2 py-0.5 text-[10px] rounded-md bg-tag-bg text-text-tertiary font-medium"
            >
              {tag}
            </span>
          ))}
        </div>
      </div>
    </Link>
  );
}
