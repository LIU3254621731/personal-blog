import fs from "fs";
import path from "path";
import matter from "gray-matter";

const CONTENT_ROOT = path.join(process.cwd(), "content");

export interface MdxFile {
  slug: string;
  frontmatter: Record<string, unknown>;
  content: string;
  filePath: string;
}

export interface MdxCategory {
  name: string;
  slug: string;
  files: MdxFile[];
}

/**
 * Get all MDX/MD files from a directory, sorted by date (desc).
 */
export function getMdxFiles(
  dir: string,
  options?: { includeDrafts?: boolean }
): MdxFile[] {
  const dirPath = path.join(CONTENT_ROOT, dir);
  if (!fs.existsSync(dirPath)) return [];

  const files = walkDir(dirPath, ".md");

  return files
    .map((filePath) => {
      const raw = fs.readFileSync(filePath, "utf-8");
      const { data, content } = matter(raw);
      const slug = path
        .basename(filePath, path.extname(filePath))
        .replace(/\s+/g, "-")
        .toLowerCase();

      return {
        slug,
        frontmatter: data,
        content: content.trim(),
        filePath,
      };
    })
    .filter((f) => {
      if (options?.includeDrafts) return true;
      return f.frontmatter.draft !== true;
    })
    .sort((a, b) => {
      const da = (a.frontmatter.date as string) ?? "";
      const db = (b.frontmatter.date as string) ?? "";
      return db.localeCompare(da);
    });
}

/**
 * Get a single MDX file by slug.
 */
export function getMdxBySlug(dir: string, slug: string): MdxFile | null {
  const files = getMdxFiles(dir, { includeDrafts: true });
  return files.find((f) => f.slug === slug) ?? null;
}

/**
 * Get all slugs in a directory (for generateStaticParams).
 */
export function getAllMdxSlugs(dir: string): string[] {
  return getMdxFiles(dir, { includeDrafts: true }).map((f) => f.slug);
}

/**
 * Group MDX files in a directory by a frontmatter key (e.g. "category").
 */
export function getMdxCategories(dir: string, key: string = "category"): MdxCategory[] {
  const files = getMdxFiles(dir);
  const map = new Map<string, MdxFile[]>();

  for (const f of files) {
    const cat = (f.frontmatter[key] as string) ?? "未分类";
    if (!map.has(cat)) map.set(cat, []);
    map.get(cat)!.push(f);
  }

  return Array.from(map.entries()).map(([name, files]) => ({
    name,
    slug: name.toLowerCase().replace(/\s+/g, "-"),
    files,
  }));
}

/**
 * Get subdirectories in a content directory (each = a category).
 */
export function getMdxSubdirs(dir: string): string[] {
  const dirPath = path.join(CONTENT_ROOT, dir);
  if (!fs.existsSync(dirPath)) return [];

  return fs
    .readdirSync(dirPath, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => d.name)
    .sort();
}

function walkDir(dirPath: string, ext: string): string[] {
  const results: string[] = [];
  if (!fs.existsSync(dirPath)) return results;

  const entries = fs.readdirSync(dirPath, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dirPath, entry.name);
    if (entry.isDirectory()) {
      results.push(...walkDir(fullPath, ext));
    } else if (entry.isFile() && (entry.name.endsWith(ext) || entry.name.endsWith(".mdx"))) {
      results.push(fullPath);
    }
  }
  return results;
}
