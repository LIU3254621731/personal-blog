import { notFound } from "next/navigation";
import Link from "next/link";
import type { Metadata } from "next";
import { getProjectById } from "@/lib/db";
import { ExternalLink, GitFork, ArrowLeft } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const STATUS_MAP: Record<string, { label: string; className: string }> = {
  planning: { label: "Planning", className: "status-planning" },
  building: { label: "Building", className: "status-building" },
  testing: { label: "Testing", className: "status-testing" },
  released: { label: "Released", className: "status-released" },
  archived: { label: "Archived", className: "status-archived" },
};

interface Props {
  params: Promise<{ id: string }>;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { id } = await params;
  const project = getProjectById(id);
  if (!project) return { title: "项目未找到" };
  return {
    title: project.title,
    description: project.description,
  };
}

export default async function ProjectDetailPage({ params }: Props) {
  const { id } = await params;
  const project = getProjectById(id);
  if (!project) notFound();

  const s = STATUS_MAP[project.status] ?? STATUS_MAP.building;

  return (
    <div className="mx-auto max-w-3xl px-6">
      {/* Back link */}
      <Link
        href="/projects"
        className="inline-flex items-center gap-2 text-sm text-text-tertiary hover:text-accent transition-colors mb-8"
      >
        <ArrowLeft size={16} />
        返回项目列表
      </Link>

      {/* Header */}
      <section className="mb-12 animate-fade-up">
        <div className="flex items-center gap-3 mb-4">
          <span
            className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium ${s.className}`}
          >
            <span className="w-1.5 h-1.5 rounded-full bg-current" />
            {s.label}
          </span>
          <span className="text-xs text-text-tertiary uppercase tracking-wider">
            {project.category}
          </span>
        </div>

        <h1 className="font-display text-3xl sm:text-4xl font-semibold tracking-tight mb-4">
          {project.title}
        </h1>

        <p className="text-text-secondary text-lg leading-relaxed mb-6">
          {project.description}
        </p>

        {/* Action links */}
        <div className="flex items-center gap-3 mb-6">
          {project.githubUrl && (
            <a
              href={project.githubUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl border border-border-medium text-sm text-text-secondary hover:text-accent hover:border-accent/30 transition-all"
            >
              <GitFork size={15} />
              View Source
            </a>
          )}
          {project.demoUrl && (
            <a
              href={project.demoUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-accent text-white text-sm font-medium hover:bg-accent/90 transition-colors"
            >
              <ExternalLink size={15} />
              Live Demo
            </a>
          )}
        </div>

        {/* Tags */}
        <div className="flex flex-wrap gap-2">
          {project.tags.map((tag) => (
            <span
              key={tag}
              className="px-3 py-1 text-xs rounded-full bg-tag-bg text-text-tertiary font-medium"
            >
              {tag}
            </span>
          ))}
        </div>
      </section>

      {/* Content sections */}
      <div className="prose-custom mb-20">
        {/* If description is markdown, render it */}
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {`## 项目介绍\n\n${project.description}\n\n## 背景与目标\n\n该项目旨在解决实际问题，提供高质量的解决方案。详细信息请查看 GitHub 仓库。`}
        </ReactMarkdown>
      </div>

      {/* Metadata footer */}
      <div className="border-t border-border-light pt-6 pb-20 text-xs text-text-tertiary space-y-1">
        <p>创建时间：{project.createdAt}</p>
        <p>更新时间：{project.updatedAt}</p>
        <p>排序权重：{project.sortOrder}</p>
      </div>
    </div>
  );
}
