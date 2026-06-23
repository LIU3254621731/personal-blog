import { getProjects } from "@/lib/db";

export default function ProjectsPage() {
  const projects = getProjects();
  const categories = [...new Set(projects.map((p) => p.category))];

  return (
    <div className="mx-auto max-w-3xl px-6">
      <header className="mb-16 animate-fade-up">
        <p className="font-mono text-[11px] tracking-[0.2em] text-text-tertiary uppercase mb-4">项目</p>
        <h1 className="font-display text-4xl sm:text-5xl font-semibold tracking-tight">我所构建的一些东西。</h1>
      </header>
      {categories.map((category) => (
        <section key={category} className="mb-14 stagger">
          <h2 className="font-mono text-[11px] tracking-[0.2em] text-text-tertiary uppercase mb-5">{category}</h2>
          <div className="space-y-3">
            {projects.filter((p) => p.category === category).map((project) => (
              <div key={project.id} id={project.id}>
                <div className="glass rounded-2xl p-6 transition-all duration-400 glass-card-hover scroll-mt-28">
                  <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 mb-2">
                    <h3 className="font-display text-lg leading-snug">{project.title}</h3>
                    {project.featured && <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-text-tertiary px-2 py-0.5 rounded-full bg-black/3 self-start shrink-0">精选</span>}
                  </div>
                  <p className="text-sm text-text-secondary leading-relaxed mb-4">{project.description}</p>
                  <div className="flex flex-wrap gap-1.5">{project.tags.map((tag) => (<span key={tag} className="px-2.5 py-0.5 text-[11px] rounded-full bg-black/3 text-text-tertiary">{tag}</span>))}</div>
                </div>
              </div>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
