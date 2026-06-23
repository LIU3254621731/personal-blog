"use client";

import { motion } from "framer-motion";
import { siteConfig } from "@/data/site";
import { GlassCard } from "@/components/ui/glass-card";

const skills = [
  {
    category: "AI 与机器学习",
    items: ["PyTorch", "计算机视觉", "元学习", "证据深度学习", "NLP/LLM", "Stable Diffusion"],
  },
  {
    category: "系统与语言",
    items: ["Rust", "C++", "Python", "TypeScript", "C#", "C (嵌入式)"],
  },
  {
    category: "框架与工具",
    items: ["Next.js", "React", "Tauri", "PyQt5", "Bazel", "MediaPipe", "Unity/Godot"],
  },
  {
    category: "领域",
    items: ["rPPG/生物医学", "3D 视觉", "嵌入式系统", "游戏引擎", "全栈 Web"],
  },
];

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-3xl px-6">
      <motion.header
        className="mb-16"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.22, 0.61, 0.36, 1] }}
      >
        <p className="font-mono text-[11px] tracking-[0.2em] text-text-tertiary uppercase mb-4">
          关于
        </p>
        <h1 className="font-display text-4xl sm:text-5xl font-semibold tracking-tight mb-6">
          你好，我是 {siteConfig.author}。
        </h1>
        <div className="space-y-4 text-text-secondary leading-relaxed max-w-xl">
          <p>
            我在 AI 研究、系统工程和产品设计的交叉领域工作。从学术深度学习论文到
            生产级桌面应用和全栈 Web 平台，我享受将复杂技术转化为优雅产品的过程。
          </p>
          <p className="text-text-primary">
            目前专注于生理感知（rPPG）、3D 计算机视觉和 AI 原生开发工具。
            我相信最好的技术会隐入背景 —— 无形、可靠且优美。
          </p>
          <p>
            不写代码的时候，我在研究游戏引擎架构、捣鼓 3D 打印，
            或者阅读计算摄影相关的论文。
          </p>
        </div>
      </motion.header>

      <motion.section
        className="mb-24"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        <h2 className="font-display text-2xl font-semibold mb-8">技能与专长</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          {skills.map((group, i) => (
            <GlassCard key={group.category} hover={false} delay={i * 0.06}>
              <h3 className="font-mono text-[11px] tracking-[0.2em] text-text-tertiary uppercase mb-3">
                {group.category}
              </h3>
              <div className="flex flex-wrap gap-1.5">
                {group.items.map((skill) => (
                  <span
                    key={skill}
                    className="px-2.5 py-1 text-[11px] rounded-lg bg-black/3 text-text-secondary"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </GlassCard>
          ))}
        </div>
      </motion.section>

      <motion.section
        className="mb-16"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <h2 className="font-display text-2xl font-semibold mb-6">联系我</h2>
        <p className="text-text-secondary leading-relaxed">
          始终对有趣的合作和对话保持开放。欢迎通过{" "}
          <a
            href={siteConfig.social.github}
            className="text-accent-warm underline underline-offset-4 hover:text-text-primary transition-colors"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub
          </a>
          {" "}或邮件联系。
        </p>
      </motion.section>
    </div>
  );
}
