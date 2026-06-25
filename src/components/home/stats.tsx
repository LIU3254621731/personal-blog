import { getProjects, getPosts, getLearningActivities } from "@/lib/db";
import { FolderGit2, FileText, Calendar, BookOpen } from "lucide-react";

export function Stats() {
  const projectCount = getProjects().length;
  const postCount = getPosts().filter((p) => p.published).length;
  const activities = getLearningActivities();
  const learningDays = activities.filter((a) => a.count > 0).length;
  const totalWords = getPosts()
    .filter((p) => p.published)
    .reduce((sum, p) => sum + p.content.length, 0);

  const stats = [
    { label: "项目", value: projectCount, Icon: FolderGit2 },
    { label: "文章", value: postCount, Icon: FileText },
    { label: "学习天数", value: learningDays, Icon: Calendar },
    {
      label: "累计字数",
      value: `${Math.round(totalWords / 1000)}k`,
      Icon: BookOpen,
    },
  ];

  return (
    <section>
      <p className="font-mono text-[11px] tracking-[0.15em] text-text-tertiary uppercase mb-3">
        Stats
      </p>
      <div className="glass rounded-xl p-4">
        <div className="grid grid-cols-2 gap-3">
          {stats.map(({ label, value, Icon }) => (
            <div key={label} className="text-center">
              <span className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-accent-light dark:bg-accent-light/20 text-accent mb-1.5">
                <Icon size={14} />
              </span>
              <p className="font-display text-lg font-semibold tracking-tight text-text-primary">
                {value}
              </p>
              <p className="text-[10px] text-text-tertiary tracking-wide">
                {label}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
