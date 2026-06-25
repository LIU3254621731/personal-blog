import { getSiteConfig } from "@/lib/db";
import { RoadmapAdminControls } from "./RoadmapAdminControls";

interface RoadmapGoal {
  key: string;
  title: string;
  description: string;
  progress: number; // 0-100
  year: string;
}

function parseRoadmapGoals(config: Record<string, string>): RoadmapGoal[] {
  const goals: RoadmapGoal[] = [];
  for (const [key, value] of Object.entries(config)) {
    if (!key.startsWith("roadmap_")) continue;
    try {
      const data = JSON.parse(value);
      goals.push({
        key,
        title: data.title ?? key.replace("roadmap_", ""),
        description: data.description ?? "",
        progress: typeof data.progress === "number" ? Math.min(100, Math.max(0, data.progress)) : 0,
        year: data.year ?? "",
      });
    } catch {
      // If not valid JSON, treat as plain title
      goals.push({
        key,
        title: value,
        description: "",
        progress: 0,
        year: "",
      });
    }
  }
  // Sort by year desc, then by key
  goals.sort((a, b) => {
    if (a.year && b.year) return b.year.localeCompare(a.year);
    if (a.year) return -1;
    if (b.year) return 1;
    return a.key.localeCompare(b.key);
  });
  return goals;
}

export default function RoadmapPage() {
  const config = getSiteConfig();
  const goals = parseRoadmapGoals(config);

  return (
    <div className="mx-auto max-w-5xl px-6">
      {/* Header */}
      <section className="mb-16 animate-fade-up">
        <p className="font-mono text-[11px] tracking-[0.15em] text-text-tertiary uppercase mb-4">
          Roadmap
        </p>
        <h1 className="font-display text-4xl sm:text-5xl font-semibold tracking-tight mb-4">
          路线图
        </h1>
        <p className="text-text-secondary max-w-xl leading-relaxed">
          年度目标和公开承诺。每一项目标都是对自己的期许。
        </p>
      </section>

      {/* Admin bar */}
      <RoadmapAdminControls goals={goals} />

      {/* Goals grid */}
      {goals.length > 0 ? (
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3 mb-20">
          {goals.map((goal) => (
            <div
              key={goal.key}
              className="glass rounded-2xl p-6 transition-all duration-300 glass-card-hover flex flex-col"
            >
              {/* Year badge */}
              {goal.year && (
                <div className="mb-3">
                  <span className="inline-flex px-2.5 py-0.5 rounded-full text-[10px] font-medium bg-accent-light dark:bg-accent-light/20 text-accent">
                    {goal.year}
                  </span>
                </div>
              )}

              {/* Title */}
              <h3 className="font-display text-lg font-semibold mb-2 leading-snug">
                {goal.title}
              </h3>

              {/* Description */}
              {goal.description && (
                <p className="text-sm text-text-secondary leading-relaxed mb-5 flex-1">
                  {goal.description}
                </p>
              )}

              {/* Progress bar */}
              <div className="mt-auto mb-3">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[10px] text-text-tertiary uppercase tracking-wider">
                    进度
                  </span>
                  <span className="text-xs font-medium text-text-secondary">
                    {goal.progress}%
                  </span>
                </div>
                <div className="h-1.5 rounded-full bg-tag-bg overflow-hidden">
                  <div
                    className="h-full rounded-full bg-accent transition-all duration-500"
                    style={{ width: `${goal.progress}%` }}
                  />
                </div>
              </div>

              {/* Inline admin actions — rendered by client wrapper below */}
              <div id={`roadmap-actions-${goal.key}`} />
            </div>
          ))}
        </div>
      ) : (
        <div className="glass rounded-2xl p-12 text-center mb-20">
          <span className="text-4xl mb-4 block">🎯</span>
          <p className="text-text-secondary mb-2">还没有设定目标</p>
          <p className="text-xs text-text-tertiary">
            通过管理后台添加年度目标...
          </p>
        </div>
      )}
    </div>
  );
}
