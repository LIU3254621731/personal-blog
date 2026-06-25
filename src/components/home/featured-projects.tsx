import Link from "next/link";
import { getProjects } from "@/lib/db";

const STATUS_MAP: Record<string, { label: string; className: string }> = {
  planning: { label: "规划中", className: "status-planning" },
  building: { label: "开发中", className: "status-building" },
  testing: { label: "测试中", className: "status-testing" },
  released: { label: "已发布", className: "status-released" },
  archived: { label: "已归档", className: "status-archived" },
};

export function FeaturedProjects() {
  const projects = getProjects()
    .filter((p) => p.featured)
    .slice(0, 4);
  if (projects.length === 0) return null;

  return (
    <section>
      <div className="flex items-center justify-between mb-3">
        <p className="font-mono text-[11px] tracking-[0.15em] text-text-tertiary uppercase">
          Projects
        </p>
        <Link
          href="/projects"
          className="text-[10px] text-text-tertiary hover:text-accent transition-colors"
        >
          全部 →
        </Link>
      </div>

      <div className="space-y-2">
        {projects.map((project) => {
          const s = STATUS_MAP[project.status] ?? STATUS_MAP.building;
          return (
            <Link key={project.id} href={`/projects/${project.id}`}>
              <div className="glass rounded-xl p-3.5 transition-all duration-300 glass-card-hover">
                <div className="flex items-center gap-2 mb-1.5">
                  <span
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium ${s.className}`}
                  >
                    <span className="w-1 h-1 rounded-full bg-current" />
                    {s.label}
                  </span>
                </div>
                <h3 className="text-sm font-medium text-text-primary leading-snug mb-1">
                  {project.title}
                </h3>
                <p className="text-xs text-text-secondary line-clamp-2 leading-relaxed">
                  {project.description}
                </p>
              </div>
            </Link>
          );
        })}
      </div>
    </section>
  );
}
