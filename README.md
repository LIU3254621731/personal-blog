<h1 align="center">🏠 Wenlin Lab — Personal Blog</h1>

<p align="center">
  <strong>Personal blog & portfolio — 夏阳的数字实验室</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Next.js-16-000000?logo=nextdotjs" alt="Next.js 16">
  <img src="https://img.shields.io/badge/React-19-61DAFB?logo=react" alt="React 19">
  <img src="https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript" alt="TypeScript">
  <img src="https://img.shields.io/badge/SQLite-better--sqlite3-003B57?logo=sqlite" alt="SQLite">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## 📖 Overview

Personal blog and portfolio for **夏阳 (Xia Yang)** — built with Next.js 16 App Router. Features a clean two-column layout, full-text search, tag filtering, admin CMS with JWT auth, and Glass-morphism design.

**Live:** `https://wenlinlab.dev` (deploying via EdgeOne)

## ✨ Features

### 📝 Content
- **Blog** — Markdown posts with `rehype-highlight` syntax highlighting
- **Tag System** — Filterable tags, tag archive page
- **Full-Text Search** — Cmd+K palette searches titles + content
- **Related Posts** — Tag-overlap based recommendations
- **Reading Time** — Automatic estimation (Chinese 300 chars/min)
- **TOC Sidebar** — Sticky table of contents with scroll spy
- **RSS Feed** — Auto-generated `/feed.xml`

### 🎨 Design
- **Glass Morphism** — Apple/Linear-inspired frosted glass UI
- **Dark Mode** — System-preference detection with no flash
- **Responsive** — Mobile-first Tailwind CSS 4
- **Framer Motion** — Page transitions + staggered animations

### 🔐 Admin
- **JWT Auth** — bcrypt password + httpOnly cookie + CSRF protection
- **Dashboard** — Post/Project stats with quick actions
- **Post Editor** — Full Markdown editor with image upload
- **Project Manager** — CRUD with status tracking
- **Settings** — Site config + password change
- **Security** — Rate limiting, CSP, HSTS, CSRF, security headers

### 🏠 Homepage
- Compact hero with typewriter tagline
- Two-column layout: posts + sidebar (daily status, projects, stats)
- GitHub-style learning heatmap

## 🚀 Quick Start

```bash
# Install
npm install

# Setup auth (generates auth.config.json + .env)
node scripts/setup-env.js

# Dev server
npm run dev
# → http://localhost:3000

# Admin: click "·" in footer → enter password
```

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | Next.js 16 (App Router, Turbopack) |
| UI | React 19, Tailwind CSS 4, Framer Motion, Radix UI |
| Database | better-sqlite3 (WAL mode) |
| Auth | bcryptjs + jose (JWT) + CSRF |
| Validation | Zod |
| Content | react-markdown + remark-gfm + rehype-highlight |
| Search | Custom full-text with highlight |

## 📁 Project Structure

```
src/
├── app/
│   ├── page.tsx              # Homepage (two-column layout)
│   ├── blog/                  # Blog list, detail, tags
│   ├── projects/              # Project showcase
│   ├── admin/                 # CMS dashboard + editor
│   └── api/                   # REST API routes
├── components/
│   ├── home/                  # Hero, posts, projects, stats, heatmap
│   ├── blog/                  # TOC, scroll progress
│   ├── admin/                 # Login modal, sidebar
│   ├── search/                # Cmd+K command palette
│   └── layout/                # Navigation, footer, theme
├── lib/
│   ├── auth.ts                # JWT + bcrypt + CSRF
│   ├── db.ts                  # SQLite CRUD
│   ├── validation.ts          # Zod schemas
│   └── admin-fetch.ts         # CSRF-aware fetch wrapper
└── proxy.ts                   # Rate limiting + security headers
```

## 🔒 Security

- ✅ Rate limiting (60 req/60s per IP, 5 for auth)
- ✅ CSRF double-submit cookie
- ✅ CSP headers (next.config.ts)
- ✅ HSTS (production), X-Frame-Options, X-Content-Type-Options
- ✅ Zod input validation on all API routes
- ✅ File upload type/size restrictions
- ✅ httpOnly + secure (production) cookies

## 📝 License

MIT
