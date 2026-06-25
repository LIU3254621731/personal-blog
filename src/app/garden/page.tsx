import { getGardenEntries } from "@/lib/db";
import { GardenCard } from "@/components/garden/garden-card";
import { GardenAdminBar, GardenAdminActions } from "@/components/admin/ProjectGardenAdminControls";

const STAGES = ["seedling", "bud", "evergreen"] as const;
const CATEGORIES = ["thought", "inspiration", "observation", "startup", "product"] as const;
const CATEGORY_NAMES: Record<string, string> = {
  thought: "思考", inspiration: "灵感", observation: "观察", startup: "创业", product: "产品",
};

export default function GardenPage() {
  const dbEntries = getGardenEntries();

  return (
    <div className="mx-auto max-w-3xl px-6 py-12">
      <header className="mb-10">
        <p className="font-mono text-[11px] tracking-[0.15em] text-text-tertiary uppercase mb-3">Garden</p>
        <h1 className="font-display text-3xl font-semibold text-text-primary mb-2">数字花园</h1>
        <p className="text-sm text-text-secondary">灵感、思考与创业观察</p>
      </header>

      <GardenAdminBar />

      {dbEntries.length === 0 ? (
        <p className="text-text-tertiary text-center py-16">花园暂空。登录后可播种新想法。</p>
      ) : (
        <div className="space-y-3">
          {dbEntries.map((entry) => (
            <div key={entry.id}>
              <GardenCard entry={entry} />
              <div className="mt-1 flex justify-end">
                <GardenAdminActions entryId={entry.id} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
