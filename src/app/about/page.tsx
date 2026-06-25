"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { siteConfig } from "@/data/site";
import { ResumeDownloadModal } from "@/components/admin/ResumeDownloadModal";
import {
  GitFork,
  Mail,
  Download,
  Code2,
  Cpu,
  Layers,
  Wrench,
  GraduationCap,
  Briefcase,
  MapPin,
} from "lucide-react";

const skills = [
  {
    category: "AI 与机器学习",
    icon: <Cpu size={16} />,
    items: [
      { name: "PyTorch", level: 90 },
      { name: "计算机视觉", level: 85 },
      { name: "NLP / LLM", level: 75 },
      { name: "Stable Diffusion", level: 70 },
      { name: "元学习", level: 80 },
    ],
  },
  {
    category: "编程语言",
    icon: <Code2 size={16} />,
    items: [
      { name: "Python", level: 92 },
      { name: "TypeScript", level: 82 },
      { name: "Rust", level: 65 },
      { name: "C++", level: 70 },
      { name: "C#", level: 60 },
    ],
  },
  {
    category: "框架与工具",
    icon: <Layers size={16} />,
    items: [
      { name: "Next.js / React", level: 85 },
      { name: "Tauri", level: 65 },
      { name: "PyQt5", level: 75 },
      { name: "MediaPipe", level: 78 },
      { name: "Unity / Godot", level: 60 },
    ],
  },
  {
    category: "领域专长",
    icon: <Wrench size={16} />,
    items: [
      { name: "rPPG / 生物医学", level: 85 },
      { name: "3D 视觉", level: 72 },
      { name: "嵌入式系统", level: 65 },
      { name: "全栈 Web", level: 80 },
      { name: "游戏引擎", level: 55 },
    ],
  },
];

const timeline = [
  {
    year: "2026",
    title: "考研 & AI 深入学习",
    description: "系统复习数学、英语、专业课，同时深入学习 Transformer、RAG、AI Agent 等前沿技术。",
    icon: <GraduationCap size={16} />,
  },
  {
    year: "2025",
    title: "全栈 & 开源项目",
    description: "构建 VtuberHub、LLM Wiki 等全栈项目，参与开源社区维护。",
    icon: <Code2 size={16} />,
  },
  {
    year: "2024",
    title: "AI 研究与产品化",
    description: "复现 rPPG 论文，将学术研究转化为桌面应用产品。探索大语言模型应用。",
    icon: <Cpu size={16} />,
  },
  {
    year: "2023",
    title: "本科深度学习起点",
    description: "开始系统学习深度学习，从 NumPy 手写框架到 PyTorch 项目实战。",
    icon: <Briefcase size={16} />,
  },
];

export default function AboutPage() {
  const [resumeOpen, setResumeOpen] = useState(false);

  return (
    <div className="mx-auto max-w-5xl px-6">
      {/* Header / Bio */}
      <motion.section
        className="mb-20"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.22, 0.61, 0.36, 1] }}
      >
        <p className="font-mono text-[11px] tracking-[0.15em] text-text-tertiary uppercase mb-4">
          About
        </p>
        <h1 className="font-display text-4xl sm:text-5xl font-semibold tracking-tight mb-6">
          你好，我是 {siteConfig.author}。
        </h1>
        <div className="space-y-4 text-text-secondary leading-relaxed max-w-2xl">
          <p className="text-lg text-text-primary">
            AI Developer · Student · Builder — 在 AI 研究、系统工程和产品设计的交叉领域工作。
          </p>
          <p>
            从学术深度学习论文到生产级桌面应用和全栈 Web 平台，
            我享受将复杂技术转化为优雅产品的过程。
            目前专注于生理感知（rPPG）、3D 计算机视觉和 AI 原生开发工具。
          </p>
          <p>
            我相信最好的技术会隐入背景 —— 无形、可靠且优美。
            不写代码的时候，我在研究游戏引擎架构、捣鼓 3D 打印，或者阅读计算摄影相关的论文。
          </p>
        </div>

        {/* Contact buttons */}
        <div className="flex flex-wrap items-center gap-3 mt-8">
          <a
            href={siteConfig.social.github}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl border border-border-medium text-sm text-text-secondary hover:text-accent hover:border-accent/30 transition-all"
          >
            <GitFork size={16} />
            GitHub
          </a>
          <a
            href={`mailto:${siteConfig.social.email}`}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl border border-border-medium text-sm text-text-secondary hover:text-accent hover:border-accent/30 transition-all"
          >
            <Mail size={16} />
            Email
          </a>
          <button
            onClick={() => setResumeOpen(true)}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-accent text-white text-sm font-medium hover:bg-accent/90 transition-colors"
          >
            <Download size={16} />
            简历下载
          </button>
        </div>
      </motion.section>

      <ResumeDownloadModal open={resumeOpen} onClose={() => setResumeOpen(false)} />

      {/* Skills */}
      <motion.section
        className="mb-20"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.5 }}
      >
        <h2 className="font-display text-2xl font-semibold mb-8">技能与专长</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          {skills.map((group, i) => (
            <div key={group.category} className="glass rounded-2xl p-6">
              <div className="flex items-center gap-2.5 mb-5">
                <span className="flex items-center justify-center w-8 h-8 rounded-lg bg-accent-light dark:bg-accent-light/20 text-accent">
                  {group.icon}
                </span>
                <h3 className="font-display text-sm font-semibold">
                  {group.category}
                </h3>
              </div>
              <div className="space-y-3">
                {group.items.map((skill) => (
                  <div key={skill.name}>
                    <div className="flex justify-between text-xs mb-1.5">
                      <span className="text-text-secondary font-medium">
                        {skill.name}
                      </span>
                      <span className="text-text-tertiary">{skill.level}%</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-tag-bg overflow-hidden">
                      <motion.div
                        className="h-full rounded-full bg-accent/60"
                        initial={{ width: 0 }}
                        whileInView={{ width: `${skill.level}%` }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.8, delay: i * 0.1, ease: "easeOut" }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </motion.section>

      {/* Timeline */}
      <motion.section
        className="mb-20"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.5 }}
      >
        <h2 className="font-display text-2xl font-semibold mb-8">成长时间线</h2>
        <div className="relative">
          {/* Vertical line */}
          <div className="absolute left-6 top-0 bottom-0 w-px bg-border-medium hidden sm:block" />

          <div className="space-y-6">
            {timeline.map((item, i) => (
              <motion.div
                key={item.year}
                className="relative pl-16"
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: i * 0.1 }}
              >
                {/* Dot */}
                <div className="absolute left-4 w-5 h-5 rounded-full bg-accent border-4 border-bg-primary dark:border-bg-primary hidden sm:flex items-center justify-center">
                  <span className="text-white text-[8px]">●</span>
                </div>

                <div className="glass rounded-2xl p-5">
                  <div className="flex items-center gap-2.5 mb-2">
                    <span className="flex items-center justify-center w-6 h-6 rounded bg-accent-light dark:bg-accent-light/20 text-accent">
                      {item.icon}
                    </span>
                    <span className="text-sm font-semibold">{item.year}</span>
                    <span className="text-sm text-text-secondary">
                      — {item.title}
                    </span>
                  </div>
                  <p className="text-sm text-text-tertiary leading-relaxed">
                    {item.description}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.section>

      {/* Contact */}
      <motion.section
        className="mb-20 pb-10"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.5 }}
      >
        <div className="glass rounded-2xl p-8 text-center">
          <h2 className="font-display text-2xl font-semibold mb-3">保持联系</h2>
          <p className="text-text-secondary mb-6 max-w-md mx-auto">
            始终对有趣的合作和对话保持开放。欢迎通过 GitHub 或邮件联系我。
          </p>
          <div className="flex items-center justify-center gap-4">
            <a
              href={siteConfig.social.github}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-tag-bg text-sm text-text-secondary hover:text-accent hover:bg-accent-light dark:hover:bg-accent-light/20 transition-all"
            >
              <GitFork size={16} />
              GitHub
            </a>
            <a
              href={`mailto:${siteConfig.social.email}`}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-tag-bg text-sm text-text-secondary hover:text-accent hover:bg-accent-light dark:hover:bg-accent-light/20 transition-all"
            >
              <Mail size={16} />
              QQ: {siteConfig.social.email}
            </a>
            <a
              href={`mailto:${siteConfig.social.emailGmail}`}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-tag-bg text-sm text-text-secondary hover:text-accent hover:bg-accent-light dark:hover:bg-accent-light/20 transition-all"
            >
              <Mail size={16} />
              Gmail: {siteConfig.social.emailGmail}
            </a>
          </div>
        </div>
      </motion.section>
    </div>
  );
}
