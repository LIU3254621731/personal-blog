"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { siteConfig } from "@/data/site";

const TAGLINES = [
  "Building AI Products",
  "Learning in Public",
  "Exploring Future Technology",
];

export function Hero() {
  const [index, setIndex] = useState(0);
  const cycle = useCallback(() => setIndex((i) => (i + 1) % TAGLINES.length), []);

  useEffect(() => {
    const interval = setInterval(cycle, 3000);
    return () => clearInterval(interval);
  }, [cycle]);

  return (
    <section className="mb-16">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.22, 0.61, 0.36, 1] }}
      >
        {/* Name + Tagline */}
        <div className="mb-6">
          <h1 className="font-display text-3xl sm:text-4xl font-semibold tracking-tight mb-2 text-text-primary">
            夏阳
          </h1>
          <div className="flex items-center gap-2 text-text-secondary text-sm mb-3">
            <span>AI Developer</span>
            <span className="w-1 h-1 rounded-full bg-text-tertiary" />
            <span>Student</span>
            <span className="w-1 h-1 rounded-full bg-text-tertiary" />
            <span>Builder</span>
          </div>

          {/* Typewriter */}
          <div className="h-6">
            <AnimatePresence mode="wait">
              <motion.p
                key={index}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.25 }}
                className="text-accent font-medium text-sm"
              >
                {TAGLINES[index]}
              </motion.p>
            </AnimatePresence>
          </div>
        </div>

        {/* Quick Links */}
        <div className="flex items-center gap-2">
          <Link
            href="/projects"
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent/90 active:scale-[0.98] transition-all duration-200"
          >
            项目
            <ArrowRight size={14} />
          </Link>
          <Link
            href="/learning"
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg border border-border-medium text-text-secondary text-sm hover:text-text-primary hover:border-text-tertiary transition-all duration-200"
          >
            学习笔记
          </Link>
          <a
            href={siteConfig.social.github}
            target="_blank"
            rel="noreferrer"
            className="px-4 py-2 rounded-lg text-text-tertiary text-sm hover:text-text-primary hover:bg-black/3 dark:hover:bg-white/5 transition-all duration-200"
          >
            GitHub
          </a>
        </div>
      </motion.div>
    </section>
  );
}
