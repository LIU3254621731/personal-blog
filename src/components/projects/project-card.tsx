import Link from "next/link";
import type { Project } from "@/lib/db";

const STATUS_MAP: Record<string, { label: string; className: string }> = {
  planning: { label: "Planning", className: "status-planning" },
  building: { label: "Building", className: "status-building" },
  testing: { label: "Testing", className: "status-testing" },
  released: { label: "Released", className: "status-released" },
  archived: { label: "Archived", className: "status-archived" },
};

interface Props {
  project: Project;
  showStatus?: boolean;
}

export function ProjectCard({ project, showStatus = true }: Props) {
  const s = STATUS_MAP[project.status] ?? STATUS_MAP.building;

  return (
    <Link href={`/projects/${project.id}`}>
      <div className="glass rounded-2xl p-5 h-full transition-all duration-300 glass-card-hover cursor-pointer flex flex-col">
        {/* Status badge */}
        {showStatus && (
          <div className="flex items-center justify-between mb-3">
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium ${s.className}`}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-current" />
              {s.label}
            </span>
          </div>
        )}

        {/* Title + desc */}
        <h3 className="font-display text-lg font-medium mb-2 leading-snug">
          {project.title}
        </h3>
        <p className="text-sm text-text-secondary leading-relaxed line-clamp-2 mb-4 flex-1">
          {project.description}
        </p>

        {/* Category + tags */}
        <div className="flex items-center gap-2 text-[10px] text-text-tertiary mb-2">
          <span className="uppercase tracking-wider font-medium">{project.category}</span>
        </div>
        <div className="flex flex-wrap gap-1.5 mt-auto">
          {project.tags.slice(0, 4).map((tag) => (
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
