import { getPosts, getProjects, getGardenEntries, getSiteConfig } from "@/lib/db";
import type { MetadataRoute } from "next";

export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = "https://wenlinlab.cn";
  const posts = getPosts().filter((p) => p.published);
  const projects = getProjects();
  const garden = getGardenEntries();

  const staticPages = [
    { path: "", priority: 1 },
    { path: "/blog", priority: 0.8 },
    { path: "/projects", priority: 0.8 },
    { path: "/learning", priority: 0.8 },
    { path: "/garden", priority: 0.7 },
    { path: "/about", priority: 0.6 },
    { path: "/resources", priority: 0.6 },
    { path: "/roadmap", priority: 0.6 },
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

  const gardenPages = garden.map((entry) => ({
    url: `${baseUrl}/garden/${entry.slug}`,
    lastModified: new Date(entry.updatedAt),
    changeFrequency: "monthly" as const,
    priority: 0.5,
  }));

  // Include learning path detail pages
  const config = getSiteConfig();
  const learningPages: MetadataRoute.Sitemap = [];
  for (const [key] of Object.entries(config)) {
    if (key.startsWith("learning_path_")) {
      const pathId = key.replace("learning_path_", "");
      learningPages.push({
        url: `${baseUrl}/learning/${pathId}`,
        lastModified: new Date(),
        changeFrequency: "weekly" as const,
        priority: 0.6,
      });
    }
  }

  return [...staticPages, ...postPages, ...projectPages, ...gardenPages, ...learningPages];
}
