"use client";

import { useEffect, useState } from "react";

interface TocItem {
  id: string;
  text: string;
  level: number;
}

export function TableOfContents() {
  const [headings, setHeadings] = useState<TocItem[]>([]);
  const [activeId, setActiveId] = useState<string>("");

  useEffect(() => {
    // Extract headings from rendered content
    const elements = document.querySelectorAll<HTMLHeadingElement>(
      ".prose-custom h2, .prose-custom h3"
    );

    const items: TocItem[] = [];
    elements.forEach((el, idx) => {
      const id = el.id || `heading-${idx}`;
      if (!el.id) el.id = id;
      items.push({
        id,
        text: el.textContent ?? "",
        level: el.tagName === "H2" ? 2 : 3,
      });
    });
    setHeadings(items);

    // Scroll spy
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveId(entry.target.id);
          }
        }
      },
      { rootMargin: "-80px 0px -80% 0px" }
    );

    elements.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  if (headings.length < 2) return null;

  return (
    <nav className="text-sm">
      <p className="text-xs font-semibold text-text-tertiary uppercase tracking-wider mb-3">
        目录
      </p>
      <ul className="space-y-1.5 border-l-2 border-border-light">
        {headings.map((h) => (
          <li key={h.id}>
            <a
              href={`#${h.id}`}
              className={`block text-xs leading-relaxed transition-colors hover:text-accent ${
                h.level === 3 ? "pl-4" : "pl-3"
              } ${
                activeId === h.id
                  ? "text-accent font-medium border-l-2 border-accent -ml-[2px]"
                  : "text-text-tertiary"
              }`}
              onClick={(e) => {
                e.preventDefault();
                document.getElementById(h.id)?.scrollIntoView({
                  behavior: "smooth",
                  block: "start",
                });
              }}
            >
              {h.text}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
