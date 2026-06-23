import { getPosts } from "@/lib/db";
import type { MetadataRoute } from "next";

export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = "https://example.com";
  const posts = getPosts().filter(p => p.published);

  const staticPages = ["", "/blog", "/projects", "/about"].map(path => ({
    url: baseUrl + path,
    lastModified: new Date(),
    changeFrequency: "weekly" as const,
    priority: path === "" ? 1 : 0.8,
  }));

  const postPages = posts.map(post => ({
    url: baseUrl + "/blog/" + post.slug,
    lastModified: new Date(post.updatedAt),
    changeFrequency: "monthly" as const,
    priority: 0.6,
  }));

  return [...staticPages, ...postPages];
}
