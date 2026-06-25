import { NextRequest, NextResponse } from "next/server";
import { getPosts, getProjects } from "@/lib/db";

/** Extract a snippet around the first match of query in text */
function snippet(text: string, query: string, maxLen = 120): string {
  const idx = text.toLowerCase().indexOf(query.toLowerCase());
  if (idx === -1) return text.slice(0, maxLen) + (text.length > maxLen ? "..." : "");
  const start = Math.max(0, idx - 40);
  const end = Math.min(text.length, idx + query.length + 80);
  let s = text.slice(start, end);
  if (start > 0) s = "..." + s;
  if (end < text.length) s = s + "...";
  return s;
}

/** Highlight query matches in text (returns array of {text, highlight} segments) */
function highlightMatches(text: string, query: string): { text: string; highlight: boolean }[] {
  if (!query) return [{ text, highlight: false }];
  const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi");
  const parts = text.split(regex);
  return parts.map((part) => ({
    text: part,
    highlight: regex.test(part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")),
  }));
}

export async function GET(req: NextRequest) {
  const q = req.nextUrl.searchParams.get("q") ?? "";
  if (!q || q.length < 1) return NextResponse.json({ results: [] });

  const query = q.toLowerCase();

  // Search posts (title, excerpt, tags, AND content)
  const posts = getPosts()
    .filter((p) => p.published)
    .filter(
      (p) =>
        p.title.toLowerCase().includes(query) ||
        p.excerpt.toLowerCase().includes(query) ||
        p.tags.some((t) => t.toLowerCase().includes(query)) ||
        p.content.toLowerCase().includes(query),
    )
    .slice(0, 6)
    .map((p) => ({
      type: "post" as const,
      title: p.title,
      excerpt: snippet(
        p.content.toLowerCase().includes(query) ? p.content : p.excerpt,
        q,
        120,
      ),
      url: `/blog/${p.slug}`,
      highlights: highlightMatches(p.title, q),
      excerptHighlights: highlightMatches(
        snippet(p.content.toLowerCase().includes(query) ? p.content : p.excerpt, q, 120),
        q,
      ),
      group: "文章",
    }));

  // Search projects
  const projects = getProjects()
    .filter(
      (p) =>
        p.title.toLowerCase().includes(query) ||
        p.description.toLowerCase().includes(query) ||
        p.tags.some((t) => t.toLowerCase().includes(query)),
    )
    .slice(0, 4)
    .map((p) => ({
      type: "project" as const,
      title: p.title,
      excerpt: snippet(p.description, q, 120),
      url: `/projects/${p.id}`,
      highlights: highlightMatches(p.title, q),
      group: "项目",
    }));

  return NextResponse.json({ results: [...posts, ...projects] });
}
