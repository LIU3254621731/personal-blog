"use client";

import { useEffect, useState } from "react";

interface Activity {
  date: string;
  count: number;
}

const EMPTY_COLOR = "bg-black/[0.03] dark:bg-white/[0.03]";
const LEVELS = [
  "bg-accent/10",
  "bg-accent/25",
  "bg-accent/45",
  "bg-accent/70",
];

function getLevel(count: number): number {
  if (count === 0) return -1;
  if (count <= 1) return 0;
  if (count <= 2) return 1;
  if (count <= 3) return 2;
  return 3;
}

const MONTHS = ["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"];

export function Heatmap() {
  const [activities, setActivities] = useState<Activity[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/learning-activity")
      .then((r) => r.json())
      .then((data) => {
        setActivities(data.activities ?? []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <section className="mb-10">
        <div className="animate-pulse space-y-3">
          <div className="h-4 w-24 bg-tag-bg rounded" />
          <div className="h-20 bg-tag-bg rounded-xl" />
        </div>
      </section>
    );
  }

  if (activities.length === 0) return null;

  const map = new Map<string, number>();
  for (const a of activities) map.set(a.date, a.count);

  const today = new Date();
  const weeks: { date: string; count: number; level: number }[][] = [];
  let currentWeek: { date: string; count: number; level: number }[] = [];

  const startDate = new Date(today);
  startDate.setDate(startDate.getDate() - 364);
  startDate.setDate(startDate.getDate() - startDate.getDay());

  for (let i = 0; i <= 371; i++) {
    const d = new Date(startDate);
    d.setDate(d.getDate() + i);
    const dateStr = d.toISOString().split("T")[0];
    const count = map.get(dateStr) ?? 0;

    if (d.getDay() === 0 && currentWeek.length > 0) {
      weeks.push(currentWeek);
      currentWeek = [];
    }
    currentWeek.push({ date: dateStr, count, level: getLevel(count) });
    if (d >= today && d.getDay() === 6) break;
  }
  if (currentWeek.length > 0) weeks.push(currentWeek);

  // Build month labels
  const monthLabels: { label: string; weekIndex: number }[] = [];
  let lastMonth = -1;
  weeks.forEach((week, wi) => {
    if (week.length > 0) {
      const m = new Date(week[0].date).getMonth();
      if (m !== lastMonth) {
        monthLabels.push({ label: MONTHS[m], weekIndex: wi });
        lastMonth = m;
      }
    }
  });

  const CELL_SIZE = 12;
  const GAP = 3;

  return (
    <section className="mb-10">
      <div className="flex items-center justify-between mb-3">
        <p className="font-mono text-[11px] tracking-[0.15em] text-text-tertiary uppercase">
          Activity
        </p>
      </div>

      <div className="glass rounded-xl p-5 overflow-x-auto">
        <div className="inline-flex flex-col min-w-[680px]">
          {/* Month labels */}
          <div className="flex mb-1 ml-8 relative h-4">
            {monthLabels.map((m, i) => (
              <span
                key={i}
                className="absolute text-[9px] text-text-tertiary"
                style={{
                  left: `${m.weekIndex * (CELL_SIZE + GAP)}px`,
                }}
              >
                {m.label}
              </span>
            ))}
          </div>

          <div className="flex gap-[3px]">
            {/* Day labels */}
            <div className="flex flex-col gap-[3px] pr-1.5" style={{ marginTop: `${CELL_SIZE + GAP}px` }}>
              {["", "一", "", "三", "", "五", ""].map((d, i) => (
                <span key={i} className="text-[9px] text-text-tertiary leading-[12px] h-[12px]">
                  {d}
                </span>
              ))}
            </div>

            {/* Weeks */}
            <div className="flex gap-[3px]">
              {weeks.map((week, wi) => (
                <div key={wi} className="flex flex-col gap-[3px]">
                  {week.map((day, di) => (
                    <div
                      key={di}
                      title={`${day.date}: ${day.count} 次学习`}
                      className={`w-[12px] h-[12px] rounded-sm ${
                        day.level < 0 ? EMPTY_COLOR : LEVELS[day.level]
                      } transition-colors hover:ring-1 hover:ring-accent/30`}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>

          {/* Legend */}
          <div className="flex items-center gap-1.5 mt-2.5 text-[10px] text-text-tertiary ml-8">
            <span>Less</span>
            <div className={`w-[10px] h-[10px] rounded-sm ${EMPTY_COLOR}`} />
            {LEVELS.map((c, i) => (
              <div key={i} className={`w-[10px] h-[10px] rounded-sm ${c}`} />
            ))}
            <span>More</span>
          </div>
        </div>
      </div>
    </section>
  );
}
