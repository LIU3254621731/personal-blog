"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Search, FileText, FolderGit2 } from "lucide-react";

interface SearchResult {
  type: "post" | "project";
  title: string;
  excerpt: string;
  url: string;
  group: string;
  highlights?: { text: string; highlight: boolean }[];
}

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  // Toggle with Cmd+K / Ctrl+K
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      }
      if (e.key === "Escape" && open) {
        setOpen(false);
      }
    },
    [open]
  );

  useEffect(() => {
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  // Focus input on open
  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 50);
      setQuery("");
      setResults([]);
      setSelectedIdx(0);
    }
  }, [open]);

  // Search
  useEffect(() => {
    if (!query.trim()) {
      setResults([]);
      return;
    }
    setLoading(true);
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        const data = await res.json();
        setResults(data.results ?? []);
        setSelectedIdx(0);
      } catch {
        setResults([]);
      }
      setLoading(false);
    }, 200);
    return () => clearTimeout(timer);
  }, [query]);

  // Keyboard navigation
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIdx((i) => Math.min(i + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && results[selectedIdx]) {
      e.preventDefault();
      router.push(results[selectedIdx].url);
      setOpen(false);
    }
  };

  const typeIcons: Record<string, React.ReactNode> = {
    post: <FileText size={15} />,
    project: <FolderGit2 size={15} />,
  };

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="fixed inset-0 z-[100] bg-black/20 backdrop-blur-sm"
            onClick={() => setOpen(false)}
          />

          {/* Dialog */}
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: -10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: -10 }}
            transition={{ duration: 0.2, ease: [0.22, 0.61, 0.36, 1] }}
            className="fixed top-[20%] left-1/2 -translate-x-1/2 z-[101] w-full max-w-lg"
          >
            <div className="glass-strong rounded-2xl overflow-hidden shadow-lg">
              {/* Input */}
              <div className="flex items-center gap-3 px-5 py-4 border-b border-border-light">
                <Search size={18} className="text-text-tertiary shrink-0" />
                <input
                  ref={inputRef}
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={onKeyDown}
                  placeholder="搜索文章、项目..."
                  className="flex-1 bg-transparent text-sm text-text-primary placeholder:text-text-tertiary outline-none"
                />
                <kbd className="text-[10px] text-text-tertiary bg-tag-bg px-1.5 py-0.5 rounded font-mono">
                  ESC
                </kbd>
              </div>

              {/* Results */}
              <div className="max-h-80 overflow-y-auto p-2">
                {loading && (
                  <div className="px-3 py-6 text-center text-sm text-text-tertiary">
                    搜索中...
                  </div>
                )}

                {!loading && query && results.length === 0 && (
                  <div className="px-3 py-6 text-center text-sm text-text-tertiary">
                    未找到结果
                  </div>
                )}

                {!loading && !query && (
                  <div className="px-3 py-6 text-center text-sm text-text-tertiary">
                    输入关键词搜索文章和项目
                  </div>
                )}

                {results.map((r, i) => (
                  <button
                    key={`${r.type}-${r.url}`}
                    className={`w-full flex items-start gap-3 px-3 py-3 rounded-xl text-left transition-colors ${
                      i === selectedIdx
                        ? "bg-accent-light dark:bg-accent-light/20"
                        : "hover:bg-tag-bg"
                    }`}
                    onClick={() => {
                      router.push(r.url);
                      setOpen(false);
                    }}
                    onMouseEnter={() => setSelectedIdx(i)}
                  >
                    <span className="text-text-tertiary mt-0.5 shrink-0">
                      {typeIcons[r.type]}
                    </span>
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-text-primary truncate">
                        {r.highlights
                          ? r.highlights.map((h, j) =>
                              h.highlight ? (
                                <mark key={j} className="bg-accent/20 text-accent rounded-sm px-0.5">
                                  {h.text}
                                </mark>
                              ) : (
                                <span key={j}>{h.text}</span>
                              ),
                            )
                          : r.title}
                      </p>
                      <p className="text-xs text-text-tertiary line-clamp-1 mt-0.5">
                        {r.excerpt}
                      </p>
                    </div>
                    <span className="text-[10px] text-text-tertiary bg-tag-bg px-1.5 py-0.5 rounded-full shrink-0 uppercase">
                      {r.group}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
