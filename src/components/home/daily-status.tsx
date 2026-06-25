import { getDailyStatus } from "@/lib/db";
import { BookOpen, Hammer, BookMarked, Lightbulb } from "lucide-react";

const items = [
  { key: "learning" as const, label: "学习", Icon: BookOpen },
  { key: "building" as const, label: "开发", Icon: Hammer },
  { key: "reading" as const, label: "阅读", Icon: BookMarked },
  { key: "thinking" as const, label: "思考", Icon: Lightbulb },
];

export function DailyStatus() {
  const status = getDailyStatus();
  if (!status) return null;

  return (
    <section>
      <p className="font-mono text-[11px] tracking-[0.15em] text-text-tertiary uppercase mb-3">
        Today I Am
      </p>
      <div className="grid grid-cols-2 gap-2">
        {items.map(({ key, label, Icon }) => {
          const value = status[key];
          return (
            <div
              key={key}
              className="glass rounded-xl p-3.5 transition-all duration-300 glass-card-hover"
            >
              <div className="flex items-center gap-2 mb-1.5">
                <span className="flex items-center justify-center w-7 h-7 rounded-md bg-accent-light dark:bg-accent-light/20 text-accent">
                  <Icon size={13} />
                </span>
                <span className="text-[10px] text-text-tertiary font-medium tracking-wide">
                  {label}
                </span>
              </div>
              <p className="text-xs text-text-secondary leading-relaxed font-medium line-clamp-2">
                {value || "—"}
              </p>
            </div>
          );
        })}
      </div>
    </section>
  );
}
