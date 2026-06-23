"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { type ReactNode } from "react";

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  hover?: boolean;
  delay?: number;
}

export function GlassCard({
  children,
  className,
  hover = true,
  delay = 0,
}: GlassCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{
        duration: 0.55,
        ease: [0.22, 0.61, 0.36, 1],
        delay,
      }}
    >
      <div
        className={cn(
          "glass rounded-2xl p-6 transition-all duration-400",
          hover && "glass-card-hover cursor-pointer",
          className
        )}
      >
        {children}
      </div>
    </motion.div>
  );
}
