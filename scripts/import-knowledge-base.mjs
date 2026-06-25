/**
 * Import articles from E:\desktop623\advantagestudy\AI-KnowledgeBase
 * and GitHub projects into the SQLite database.
 *
 * Usage: node scripts/import-knowledge-base.mjs
 */

import Database from "better-sqlite3";
import path from "path";
import fs from "fs";
import crypto from "crypto";

const DB_PATH = path.join(process.cwd(), "data", "blog.db");
const KB_ROOT = "E:\\desktop623\\advantagestudy\\AI-KnowledgeBase";

const db = new Database(DB_PATH);
db.pragma("journal_mode = WAL");
db.pragma("busy_timeout = 5000");

// ─── Helpers ────────────────────────────────────────────

function genId() {
  return Math.random().toString(36).substring(2, 10) + Date.now().toString(36);
}

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/[《》「」""'']/g, "")
    .replace(/[^a-z0-9一-鿿]+/g, "-")
    .replace(/(^-|-$)/g, "")
    .substring(0, 100);
}

function extractTitle(filePath) {
  try {
    const content = fs.readFileSync(filePath, "utf8");
    const lines = content.split("\n");
    for (const line of lines) {
      const m = line.match(/^#\s+(.+)/);
      if (m) return m[1].trim();
    }
  } catch {}
  return path.basename(filePath, ".md");
}

function extractExcerpt(content, maxLen = 200) {
  const cleaned = content
    .replace(/^#.*$/gm, "")
    .replace(/[>|\-*`\[\]()]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  return cleaned.length > maxLen ? cleaned.substring(0, maxLen) + "..." : cleaned;
}

// ─── Tag mapping by directory ───────────────────────────

const DIR_TAG_MAP = {
  "Agent": ["AI", "Agent", "系统设计"],
  "AI-Infra": ["AI", "基础设施", "工程"],
  "Architecture": ["架构", "系统设计", "分布式"],
  "CPP": ["C++", "系统编程"],
  "LLM": ["AI", "LLM", "深度学习"],
  "MCP": ["AI", "MCP", "工具"],
  "Python": ["Python", "编程"],
  "RAG": ["AI", "RAG", "检索"],
  "Rust": ["Rust", "系统编程"],
  "VectorDB": ["AI", "向量数据库", "检索"],
  "Prompt": ["AI", "Prompt", "工程"],
  "LangGraph": ["AI", "Agent", "LangGraph"],
  "RFC": ["AI", "规范", "标准"],
  "Interview": ["面试", "求职"],
  "OfficialDocs": ["文档", "官方"],
  "Papers": ["论文", "AI", "研究"],
  "SourceCode": ["源码", "工程"],
  "Troubleshooting": ["排错", "工程"],
};

// ─── Import Articles ─────────────────────────────────────

function importArticles() {
  console.log("\n📚 Importing articles from AI-KnowledgeBase...\n");

  const categories = fs.readdirSync(KB_ROOT, { withFileTypes: true })
    .filter(d => d.isDirectory())
    .map(d => d.name);

  let total = 0;
  const learningPaths = new Map(); // category -> { tags, count }

  for (const cat of categories) {
    const catDir = path.join(KB_ROOT, cat);
    const tags = DIR_TAG_MAP[cat] || [cat, "AI"];

    // Find all .md files recursively
    const mdFiles = [];
    function walk(dir) {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        if (entry.isDirectory()) walk(path.join(dir, entry.name));
        else if (entry.name.endsWith(".md")) mdFiles.push(path.join(dir, entry.name));
      }
    }
    walk(catDir);

    if (mdFiles.length === 0) {
      console.log(`  ${cat}: 0 files, skipping`);
      continue;
    }

    console.log(`  ${cat}: ${mdFiles.length} files`);

    for (const filePath of mdFiles) {
      try {
        const content = fs.readFileSync(filePath, "utf8");
        if (content.trim().length < 100) continue; // skip empty/short files

        const title = extractTitle(filePath);
        const relPath = path.relative(KB_ROOT, filePath);
        const slug = slugify(title) || slugify(path.basename(filePath, ".md"));

        // Check if slug already exists
        const existing = db.prepare("SELECT id FROM posts WHERE slug = ?").get(slug);
        if (existing) {
          console.log(`    ⏭  ${title} (slug exists)`);
          continue;
        }

        const excerpt = extractExcerpt(content);
        const now = new Date().toISOString();
        const id = genId();

        db.prepare(
          `INSERT INTO posts (id, title, slug, content, excerpt, tags, published, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)`
        ).run(id, title, slug, content, excerpt, JSON.stringify(tags), now, now);

        total++;
      } catch (e) {
        console.error(`    ❌ ${filePath}: ${e.message?.substring(0, 80)}`);
      }
    }

    // Track learning path info
    learningPaths.set(cat, {
      tags: tags,
      count: mdFiles.length,
      pathName: cat,
      pathTags: [cat, ...tags].filter((v, i, a) => a.indexOf(v) === i),
    });
  }

  console.log(`\n  ✅ Imported ${total} articles total`);

  // Create learning paths
  console.log("\n🧭 Creating learning paths...\n");

  const pathOrder = [
    "LLM", "RAG", "Agent", "AI-Infra", "Architecture",
    "VectorDB", "MCP", "LangGraph", "Prompt", "CPP", "Python",
    "Rust", "Troubleshooting", "Interview", "Papers", "OfficialDocs",
    "SourceCode", "RFC"
  ];

  const pathIcons = {
    "LLM": "Brain", "RAG": "Database", "Agent": "Bot",
    "AI-Infra": "Server", "Architecture": "GitBranch",
    "VectorDB": "Database", "MCP": "Wrench", "LangGraph": "GitMerge",
    "Prompt": "MessageSquare", "CPP": "Terminal", "Python": "Terminal",
    "Rust": "Cog", "Troubleshooting": "Bug", "Interview": "UserCheck",
    "Papers": "BookOpen", "OfficialDocs": "Book", "SourceCode": "Code",
    "RFC": "FileText"
  };

  for (const [i, cat] of pathOrder.entries()) {
    const info = learningPaths.get(cat);
    if (!info || info.count === 0) continue;

    const key = `learning_path_${cat.toLowerCase().replace(/[^a-z0-9]/g, "_")}`;
    const pathData = {
      name: cat,
      description: `${info.pathTags.join(" / ")} 相关文章合集 (${info.count} 篇)`,
      icon: pathIcons[cat] || "BookOpen",
      tags: info.pathTags,
      order: i,
    };

    db.prepare("INSERT OR REPLACE INTO site_config (key, value) VALUES (?, ?)")
      .run(key, JSON.stringify(pathData));
    console.log(`  ✅ ${cat} → ${key} (${info.count} articles)`);
  }

  return total;
}

// ─── Import GitHub Projects ───────────────────────────────

function importGitHubProjects() {
  console.log("\n🔗 Importing GitHub projects...\n");

  const projects = [
    { name: "meta-learning-rppg", category: "AI / 计算机视觉", tags: ["Python", "PyTorch", "MediaPipe", "MAML", "计算机视觉"], status: "building", featured: false },
    { name: "VtuberHub", category: "桌面应用 / 3D", tags: ["C#", "Godot", "Unity", "WPF", "MediaPipe", "3D"], status: "building", featured: false },
    { name: "llmwiki_dpk", category: "全栈 / AI", tags: ["Rust", "Tauri", "TypeScript", "LLM", "RAG"], status: "building", featured: false },
    { name: "rppg_heart_rate_monitor", category: "AI / 计算机视觉", tags: ["Python", "PyQt5", "MediaPipe", "OpenCV", "rPPG"], status: "building", featured: false },
    { name: "window_face_multithreading", category: "嵌入式 / 性能", tags: ["C++", "OpenCV", "RK3588", "NPU", "ARM"], status: "building", featured: false },
    { name: "GoogleMediapipePackageDll", category: "系统 / 底层", tags: ["C++", "Bazel", "MediaPipe", "DLL", "Windows"], status: "released", featured: true },
    { name: "douyin-downloader", category: "工具 / 爬虫", tags: ["Python", "aiohttp", "抖音", "下载器"], status: "released", featured: false },
    { name: "promotmaster", category: "全栈 / AI", tags: ["TypeScript", "React", "FastAPI", "Gemini", "AI"], status: "building", featured: false },
    { name: "vicinae-main", category: "桌面应用", tags: ["C++", "桌面应用", "启动器"], status: "released", featured: false },
    { name: "snake-c", category: "趣味项目", tags: ["C++", "OpenCV", "MediaPipe", "游戏"], status: "released", featured: false },
    { name: "siderbar", category: "桌面应用", tags: ["C#", "WinUI3", "Windows"], status: "building", featured: false },
    { name: "yanxiaomap", category: "全栈 / Web", tags: ["Java", "SpringBoot", "Vue3", "地图"], status: "building", featured: false },
    { name: "Meta-rPPG-master", category: "AI / 计算机视觉", tags: ["Python", "PyTorch", "rPPG", "元学习"], status: "testing", featured: true },
    { name: "Deep-rPPG", category: "AI / 计算机视觉", tags: ["Python", "PyTorch", "rPPG", "深度学习"], status: "testing", featured: false },
    { name: "rPPG-Toolbox", category: "AI / 计算机视觉", tags: ["Python", "PyTorch", "rPPG", "工具箱"], status: "testing", featured: false },
    { name: "MetaPhys", category: "AI / 计算机视觉", tags: ["Python", "PyTorch", "rPPG", "元学习"], status: "testing", featured: false },
    { name: "codex-bridge-main", category: "工具 / LLM", tags: ["JavaScript", "Node.js", "OpenAI", "API"], status: "released", featured: false },
    { name: "Ac-Wiki", category: "内容 / 教育", tags: ["HTML", "教育", "Wiki"], status: "building", featured: false },
    { name: "TrendRadar", category: "工具 / 舆情", tags: ["Python", "舆情", "监控", "AI"], status: "released", featured: false },
  ];

  const repoData = {
    "meta-learning-rppg": { desc: "三支路rPPG架构：自监督预训练 + MAML元学习 + 证据深度学习，从面部视频非接触式估计心率", github: "https://github.com/LIU3254621731/meta-learning-rppg" },
    "VtuberHub": { desc: "跨WPF、Godot、Unity的完整虚拟主播工具集，基于MediaPipe实现实时3D角色驱动", github: "https://github.com/LIU3254621731/VtuberHub" },
    "llmwiki_dpk": { desc: "基于Tauri + Rust构建的AI原生知识管理系统，集成LLM检索增强生成与语义搜索", github: "https://github.com/LIU3254621731/llmwiki_dpk" },
    "rppg_heart_rate_monitor": { desc: "基于PyQt5、MediaPipe与OpenCV的非接触式实时心率监测桌面应用", github: "https://github.com/LIU3254621731/rppg_heart_rate_monitor" },
    "window_face_multithreading": { desc: "RK3588 ARM平台上的高性能rPPG心率监测，NPU加速人脸检测(C++17 + OpenCV + RKNN)", github: "https://github.com/LIU3254621731/window_face_multithreading" },
    "GoogleMediapipePackageDll": { desc: "将Google MediaPipe全手跟踪封装为可复用Windows DLL——Bazel定制构建", github: "https://github.com/LIU3254621731/GoogleMediapipePackageDll" },
    "douyin-downloader": { desc: "功能丰富的抖音/TikTok批量下载器：视频、图片、音乐、直播录制(Python + aiohttp)", github: "https://github.com/LIU3254621731/douyin-downloader" },
    "promotmaster": { desc: "PromptMaster Pro - AI提示词管理与优化工具(React 19 + FastAPI + Gemini)", github: "https://github.com/LIU3254621731/promotmaster" },
    "vicinae-main": { desc: "高性能原生C++桌面命令面板/启动器", github: "https://github.com/LIU3254621731/vicinae-main" },
    "snake-c": { desc: "手势控制贪吃蛇游戏，使用MediaPipe手部跟踪(C++ + OpenCV)", github: "https://github.com/LIU3254621731/snake-c" },
    "siderbar": { desc: "Windows 11流畅设计工作流Dock栏(WinUI3 + MVVM)", github: "https://github.com/LIU3254621731/siderbar" },
    "yanxiaomap": { desc: "研究生院选择平台，带交互式地图(Spring Boot + Vue3 + AMap + JWT)", github: "https://github.com/LIU3254621731/yanxiaomap" },
    "Meta-rPPG-master": { desc: "ECCV 2020 - 使用传导元学习器的远程心率估计(PyTorch)", github: "https://github.com/LIU3254621731/Meta-rPPG-master" },
    "Deep-rPPG": { desc: "基于深度学习的远程光电容积描记(Deep-rPPG)：从视频中提取脉搏信号", github: "https://github.com/LIU3254621731/Deep-rPPG" },
    "rPPG-Toolbox": { desc: "Deep Remote PPG Toolbox (NeurIPS 2023) - 深度rPPG工具箱", github: "https://github.com/LIU3254621731/rPPG-Toolbox" },
    "MetaPhys": { desc: "MetaPhys: 非接触式生理测量的少样本自适应(ACM CHIL-2021)", github: "https://github.com/LIU3254621731/MetaPhys" },
    "codex-bridge-main": { desc: "零依赖Node.js桥: OpenAI Responses API ↔ Chat Completions API协议转换器", github: "https://github.com/LIU3254621731/codex-bridge-main" },
    "Ac-Wiki": { desc: "高等教育学社基础知识开源建设工程——青年大学习", github: "https://github.com/LIU3254621731/Ac-Wiki" },
    "TrendRadar": { desc: "AI新闻热点聚合与舆情监控工具，支持35个平台，自动推送+AI分析", github: "https://github.com/LIU3254621731/TrendRadar" },
  };

  let count = 0;
  for (const p of projects) {
    const data = repoData[p.name] || { desc: "", github: `https://github.com/LIU3254621731/${p.name}` };
    const now = new Date().toISOString();
    const id = genId();

    // Check if exists by title
    const existing = db.prepare("SELECT id FROM projects WHERE title = ?").get(p.name);
    if (existing) {
      // Update existing
      db.prepare(
        `UPDATE projects SET category=?, description=?, tags=?, status=?, github_url=?, updated_at=? WHERE id=?`
      ).run(p.category, data.desc, JSON.stringify(p.tags), p.status, data.github, now, existing.id);
      console.log(`  🔄 ${p.name} (updated)`);
    } else {
      db.prepare(
        `INSERT INTO projects (id, title, category, description, tags, featured, sort_order, status, github_url, demo_url, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      ).run(id, p.name, p.category, data.desc, JSON.stringify(p.tags), p.featured ? 1 : 0, count, p.status, data.github, "", now, now);
      console.log(`  ✅ ${p.name} (new)`);
    }
    count++;
  }

  console.log(`\n  ✅ ${count} projects synced`);
}

// ─── Main ─────────────────────────────────────────────────

console.log("🚀 Starting data import...");
importArticles();
importGitHubProjects();

console.log("\n🎉 Done! Run `npm run dev` to see the results.");
