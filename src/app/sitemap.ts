import { getPosts, getProjects } from "@/lib/db";
import type { MetadataRoute } from "next";

export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = "https://wenlinlab.dev";
  const posts = getPosts().filter((p) => p.published);
  const projects = getProjects();

  const staticPages = [
    { path: "", priority: 1 },
    { path: "/blog", priority: 0.8 },
    { path: "/projects", priority: 0.8 },
    { path: "/learning", priority: 0.8 },
    { path: "/garden", priority: 0.7 },
    { path: "/about", priority: 0.6 },
  ].map(({ path, priority }) => ({
    url: baseUrl + path,
    lastModified: new Date(),
    changeFrequency: "weekly" as const,
    priority,
  }));

  const postPages = posts.map((post) => ({
    url: `${baseUrl}/blog/${post.slug}`,
    lastModified: new Date(post.updatedAt),
    changeFrequency: "monthly" as const,
    priority: 0.6,
  }));

  const projectPages = projects.map((project) => ({
    url: `${baseUrl}/projects/${project.id}`,
    lastModified: new Date(project.updatedAt),
    changeFrequency: "monthly" as const,
    priority: 0.6,
  }));

  return [...staticPages, ...postPages, ...projectPages];
}
