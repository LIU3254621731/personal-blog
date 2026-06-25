const D = require("better-sqlite3");
const d = new D("data/blog.db");

const posts = d.prepare("SELECT title,slug,content,excerpt,tags,published,created_at,updated_at FROM posts WHERE published=1 ORDER BY created_at DESC LIMIT 8").all();
const projects = d.prepare("SELECT title,category,description,tags,featured,sort_order,status,github_url,demo_url,created_at,updated_at FROM projects ORDER BY sort_order").all();
const config = d.prepare("SELECT key,value FROM site_config WHERE key NOT LIKE 'learning_path_%'").all();
const garden = d.prepare("SELECT title,slug,content,excerpt,tags,category,stage,published,created_at,updated_at FROM garden_entries WHERE published=1 ORDER BY created_at DESC").all();
const learningPaths = d.prepare("SELECT key,value FROM site_config WHERE key LIKE 'learning_path_%'").all();

console.log(JSON.stringify({ posts, projects, config, garden, learningPaths }, null, 2));
