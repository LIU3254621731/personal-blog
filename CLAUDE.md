# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: This is Next.js 16 — APIs may differ

This project uses Next.js 16.2.9 (App Router). This version has breaking changes from earlier Next.js. Before writing any Next.js-specific code, consult the relevant guide in `node_modules/next/dist/docs/01-app/`. Key docs:

- `01-getting-started/` — project structure, layouts, data fetching, server/client components, route handlers
- `03-api-reference/04-functions/` — `after()`, `cacheLife()`, etc.
- `03-api-reference/03-file-conventions/` — metadata, layouts, pages, etc.

## Commands

```bash
npm run dev      # Start dev server (localhost:3000)
npm run build    # Production build
npm run start    # Start production server
npm run lint     # ESLint
```

## Architecture

Personal blog/CMS — single-user portfolio with a public frontend and a cookie-authenticated admin backend. All data stored in a local SQLite database (`data/blog.db`, auto-created on first access).

### Tech Stack

- **Framework:** Next.js 16.2.9 (App Router)
- **React:** 19.2.4
- **Styling:** Tailwind CSS 4 + custom CSS design tokens (glass morphism)
- **DB:** better-sqlite3 (WAL mode, file at `data/blog.db`)
- **Auth:** Cookie-based (`admin_token`), hardcoded password in `src/lib/auth.ts`
- **Content:** react-markdown (remark-gfm, rehype-highlight)
- **Animation:** framer-motion
- **Theming:** next-themes + inline `<script>` in layout to prevent flash

### Data Model (SQLite)

Three tables in `src/lib/db.ts`:

- **posts** — id, title, slug (unique), content, excerpt, tags (JSON array), published (0/1), created_at, updated_at
- **projects** — id, title, category, description, tags (JSON), featured (0/1), sort_order, created_at, updated_at
- **site_config** — key/value store for editable site metadata

Seed data is inserted automatically when the posts table is empty.

### Route Structure

| Route              | Type          | Description                                                          |
| ------------------ | ------------- | -------------------------------------------------------------------- |
| `/`                | Server        | Home: featured projects + 3 latest posts                             |
| `/blog`            | Server        | All published posts                                                  |
| `/blog/[slug]`     | Server        | Single post with Markdown; `generateStaticParams()` pre-builds slugs |
| `/projects`        | Server        | Projects grouped by category                                         |
| `/about`           | Client        | About page with skills grid (uses framer-motion)                     |
| `/admin`           | Server        | Auth guard via `isAuthenticated()` → redirects to `/admin/login`    |
| `/admin/posts`     | Client        | CRUD list, fetches from API                                          |
| `/admin/posts/new` | Client        | PostEditor in create mode                                            |
| `/admin/posts/[id]`| Server        | PostEditor in edit mode (fetches post server-side)                   |
| `/admin/projects`  | Client        | CRUD list with inline form, fetches from API                         |
| `/admin/settings`  | Client        | Editable site config form                                            |
| `/admin/login`     | Client        | Password form → sets auth cookie                                     |
| `/api/posts`       | Route handler | GET (public) / POST (auth)                                           |
| `/api/posts/[id]`  | Route handler | GET (public) / PUT (auth) / DELETE (auth)                            |
| `/api/projects`    | Route handler | GET (public) / POST (auth)                                           |
| `/api/projects/[id]`| Route handler| GET (public) / PUT (auth) / DELETE (auth)                            |
| `/api/auth/login`  | Route handler | POST — validates password, sets cookie                               |
| `/api/auth/logout` | Route handler | POST — deletes auth cookie                                          |
| `/api/site-config` | Route handler | GET (public) / PUT (auth)                                           |
| `/api/upload`      | Route handler | POST (auth) — saves image to `public/uploads/`                      |

### Key Modules

- **`src/lib/db.ts`** — All database access. Lazy-initializes SQLite on first call, creates tables, seeds if empty. Exports typed `Post` and `Project` interfaces and full CRUD functions.
- **`src/lib/auth.ts`** — Cookie-based auth helpers: `login(password)`, `logout()`, `isAuthenticated()`. Password is hardcoded (`admin123`).
- **`src/lib/utils.ts`** — `cn()` (clsx + tailwind-merge) for conditional class names; `formatDate()` using `zh-CN` locale.
- **`src/lib/reading-time.ts`** — Estimates reading time from Chinese character count (300 chars/min).
- **`src/data/site.ts`** — Static site config (nav links, author name, social links). Used by layout/navigation/footer. Note: there's also a DB-backed `site_config` table for runtime-editable settings.
- **`src/proxy.ts`** — Adds `x-pathname` header to every response (used by admin layout to detect current path). Next.js 16 renamed `middleware` → `proxy`.

### Design System

Custom properties defined in `globals.css` (`:root` and `.dark`) mapped to Tailwind via `@theme inline`:

- Colors: `--bg-primary/secondary`, `--text-primary/secondary/tertiary`, `--accent`, `--accent-warm`, `--border-light/medium`
- Glass classes: `.glass` (backdrop-blur, semi-transparent), `.glass-strong` (more opaque), `.glass-card-hover` (lift on hover)
- Typography: `.prose-custom` for blog content, `.font-display` with tightened letter-spacing
- Animations: `.animate-fade-up`, `.stagger` (staggered child delays), `.link-underline` (slide-in underline)

Fonts loaded in root layout: Inter (body), Playfair Display (headings), JetBrains Mono (code).

### Component Patterns

- **Public pages** use Server Components by default (`getPosts()`, `getProjects()` called directly at module level — these are synchronous SQLite reads).
- **Admin pages** are Client Components (`"use client"`) fetching from API routes with auth checks.
- **`Navigation`** uses `layoutId` from framer-motion for the animated active-nav indicator.
- **`GlassCard`** wraps content with entrance animation + glass styling.

### Authentication Flow

1. User POSTs password to `/api/auth/login`
2. Server sets `admin_token` cookie (httpOnly, 7-day expiry)
3. Admin layout checks `isAuthenticated()` server-side; redirects to `/admin/login` if unauthenticated (skips redirect when already on `/admin/login` to prevent redirect loops, detected via `x-pathname` header from proxy)
4. Write API routes check `isAuthenticated()` on each request
