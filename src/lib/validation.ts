/**
 * Zod validation schemas for API inputs.
 * Every write API route MUST validate its input through these schemas.
 */

import { z } from "zod";
import { NextResponse } from "next/server";

/**
 * Parse and validate a JSON request body against a Zod schema.
 * Returns the parsed data on success, or a 400 JSON response on failure.
 *
 * Usage in a route handler:
 *   const parsed = await validateBody(req, createPostSchema);
 *   if (parsed instanceof NextResponse) return parsed;
 */
export async function validateBody<T extends z.ZodTypeAny>(
  req: Request,
  schema: T,
): Promise<z.infer<T> | NextResponse> {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "请求体必须是有效的 JSON" }, { status: 400 });
  }

  const result = schema.safeParse(body);
  if (!result.success) {
    const issues = result.error.issues.map((i) => ({
      field: i.path.join("."),
      message: i.message,
    }));
    return NextResponse.json({ error: "输入校验失败", issues }, { status: 400 });
  }

  return result.data;
}

// ─── Auth ──────────────────────────────────────────────

export const loginSchema = z.object({
  password: z.string().min(1, "密码不能为空").max(128),
});

export const changePasswordSchema = z.object({
  currentPassword: z.string().min(1, "当前密码不能为空"),
  newPassword: z.string().min(6, "新密码至少需要 6 个字符").max(128),
});

// ─── Posts ─────────────────────────────────────────────

const slugRegex = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

export const createPostSchema = z.object({
  title: z.string().min(1, "标题不能为空").max(200),
  slug: z
    .string()
    .min(1)
    .max(120)
    .regex(slugRegex, "slug 只能包含小写字母、数字和连字符"),
  content: z.string().default(""),
  excerpt: z.string().max(500).default(""),
  tags: z.array(z.string().max(50)).max(20).default([]),
  published: z.boolean().default(true),
});

export const updatePostSchema = createPostSchema.partial();

// ─── Projects ──────────────────────────────────────────

export const createProjectSchema = z.object({
  title: z.string().min(1, "标题不能为空").max(200),
  category: z.string().max(100).default(""),
  description: z.string().max(2000).default(""),
  tags: z.array(z.string().max(50)).max(20).default([]),
  featured: z.boolean().default(false),
  sortOrder: z.number().int().min(0).max(9999).default(0),
  status: z.enum(["building", "released", "archived"]).default("building"),
  githubUrl: z.string().max(500).default(""),
  demoUrl: z.string().max(500).default(""),
});

export const updateProjectSchema = createProjectSchema.partial();

// ─── Garden ────────────────────────────────────────────

export const createGardenSchema = z.object({
  title: z.string().min(1, "标题不能为空").max(200),
  slug: z.string().min(1).max(120).regex(slugRegex),
  content: z.string().default(""),
  excerpt: z.string().max(500).default(""),
  tags: z.array(z.string().max(50)).max(20).default([]),
  category: z.enum(["thought", "observation", "project", "note"]).default("thought"),
  stage: z.enum(["seedling", "bud", "evergreen"]).default("seedling"),
  published: z.boolean().default(true),
});

export const updateGardenSchema = createGardenSchema.partial();

// ─── Site Config ───────────────────────────────────────

export const siteConfigSchema = z.record(z.string(), z.string());

// ─── Daily Status ──────────────────────────────────────

export const dailyStatusSchema = z.object({
  learning: z.string().max(200).default(""),
  building: z.string().max(200).default(""),
  reading: z.string().max(200).default(""),
  thinking: z.string().max(200).default(""),
});

// ─── Upload ────────────────────────────────────────────

/** Allowed image MIME types for uploads */
export const ALLOWED_IMAGE_TYPES = [
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
  "image/svg+xml",
] as const;

export const MAX_UPLOAD_SIZE = 5 * 1024 * 1024; // 5 MB
