import { getProjects } from "@/lib/db";
import { ProjectCard } from "@/components/projects/project-card";
import type { Project } from "@/lib/db";
import { ProjectAdminBar, ProjectAdminActions } from "@/components/admin/ProjectGardenAdminControls";

export const dynamic = "force-dynamic";

function makeHash(cat: string): string {
  return cat.toLowerCase().replace(/[^a-z0-9一-鿿]+/g, "-").replace(/(^-|-$)/g, "");
}

export default function ProjectsPage() {
  const projects = getProjects();

  // Group by category
  const grouped = new Map<string, Project[]>();
  for (const p of projects) {
    const cat = p.category || "未分类";
    if (!grouped.has(cat)) grouped.set(cat, []);
    grouped.get(cat)!.push(p);
  }
  const cats = Array.from(grouped.keys());

  return (
    <div className="mx-auto max-w-5xl px-6">
      {/* Header */}
      <section className="mb-16 animate-fade-up">
        <p className="font-mono text-[11px] tracking-[0.15em] text-text-tertiary uppercase mb-4">
          Projects
        </p>
        <h1 className="font-display text-4xl sm:text-5xl font-semibold tracking-tight mb-4">
          项目
        </h1>
        <p className="text-text-secondary max-w-xl leading-relaxed">
          我构建和参与的项目，涵盖 AI、Web 开发、研究和开源工具。
        </p>
      </section>

      {/* Category tabs — dynamic from actual data */}
      {cats.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-12">
          <a
            href="#all"
            className="px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 bg-tag-bg text-text-secondary hover:text-accent hover:bg-accent-light dark:hover:bg-accent-light/20"
          >
            全部
          </a>
          {cats.map((cat) => (
            <a
              key={cat}
              href={`#${makeHash(cat)}`}
              className="px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 bg-tag-bg text-text-secondary hover:text-accent hover:bg-accent-light dark:hover:bg-accent-light/20"
            >
              {cat}
            </a>
          ))}
        </div>
      )}

      {/* Admin bar */}
      <ProjectAdminBar />

      {/* All projects grid */}
      <div id="all" className="mb-20">
        <h2 className="font-display text-xl font-semibold mb-6">全部项目</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {projects.map((p) => (
            <div key={p.id}>
              <ProjectCard project={p} />
              <ProjectAdminActions projectId={p.id} />
            </div>
          ))}
        </div>
        {projects.length === 0 && (
          <p className="text-text-tertiary text-sm py-12 text-center">
            暂无项目。通过管理后台添加。
          </p>
        )}
      </div>

      {/* Grouped by category */}
      {Array.from(grouped.entries()).map(([cat, items]) => (
        <div key={cat} id={makeHash(cat)} className="mb-20">
          <h2 className="font-display text-xl font-semibold mb-6">{cat}</h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {items.map((p) => (
              <div key={p.id}>
                <ProjectCard project={p} />
                <ProjectAdminActions projectId={p.id} />
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
