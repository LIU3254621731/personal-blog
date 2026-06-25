"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";
import { siteConfig } from "@/data/site";
import { cn } from "@/lib/utils";
import { ThemeToggle } from "./theme-toggle";

export function Navigation() {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <header className="fixed top-0 left-0 right-0 z-50">
        <motion.nav
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, ease: [0.22, 0.61, 0.36, 1] }}
          className="glass-strong mx-auto mt-4 max-w-3xl rounded-2xl px-5 py-2.5"
        >
          <div className="flex items-center justify-between">
            <Link
              href="/"
              className="font-display text-lg tracking-tight text-text-primary hover:text-accent transition-colors duration-300"
            >
              {siteConfig.name}
            </Link>
            <div className="hidden items-center gap-0.5 sm:flex">
              {siteConfig.nav.map((item) => {
                const isActive = pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "relative rounded-xl px-4 py-2 text-sm transition-all duration-300",
                      isActive
                        ? "text-accent font-medium"
                        : "text-text-secondary hover:text-text-primary"
                    )}
                  >
                    {isActive && (
                      <motion.span
                        layoutId="nav-active"
                        className="absolute inset-1 rounded-lg bg-accent-light dark:bg-accent-light/20"
                        transition={{ type: "spring", stiffness: 400, damping: 30 }}
                      />
                    )}
                    <span className="relative z-10">{item.label}</span>
                  </Link>
                );
              })}
              <span className="ml-2 pl-2 border-l border-border-light">
                <ThemeToggle />
              </span>
            </div>
            <div className="flex items-center gap-3 sm:hidden">
              <ThemeToggle />
              <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex flex-col gap-1.5 p-2 relative z-20"
                aria-label="Toggle menu"
              >
                <motion.span
                  animate={isOpen ? { rotate: 45, y: 6.5 } : { rotate: 0, y: 0 }}
                  transition={{ duration: 0.2 }}
                  className="block h-[1.5px] w-5 bg-text-primary rounded-full"
                />
                <motion.span
                  animate={
                    isOpen ? { opacity: 0, scale: 0.5 } : { opacity: 1, scale: 1 }
                  }
                  transition={{ duration: 0.15 }}
                  className="block h-[1.5px] w-5 bg-text-primary rounded-full"
                />
                <motion.span
                  animate={isOpen ? { rotate: -45, y: -6.5 } : { rotate: 0, y: 0 }}
                  transition={{ duration: 0.2 }}
                  className="block h-[1.5px] w-5 bg-text-primary rounded-full"
                />
              </button>
            </div>
          </div>
        </motion.nav>
      </header>

      <AnimatePresence>
        {isOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="fixed inset-0 z-30 bg-black/10 backdrop-blur-sm sm:hidden"
              onClick={() => setIsOpen(false)}
            />
            <motion.div
              initial={{ opacity: 0, y: -12, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -12, scale: 0.96 }}
              transition={{ duration: 0.25, ease: [0.22, 0.61, 0.36, 1] }}
              className="fixed top-20 left-4 right-4 z-40 sm:hidden"
            >
              <div className="glass-strong rounded-2xl p-3">
                {siteConfig.nav.map((item) => {
                  const isActive = pathname === item.href;
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      onClick={() => setIsOpen(false)}
                      className={cn(
                        "block rounded-xl px-4 py-3 text-sm transition-colors duration-200",
                        isActive
                          ? "bg-accent-light dark:bg-accent-light/20 text-accent font-medium"
                          : "text-text-secondary hover:text-text-primary hover:bg-black/3 dark:hover:bg-white/5"
                      )}
                    >
                      {item.label}
                    </Link>
                  );
                })}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
