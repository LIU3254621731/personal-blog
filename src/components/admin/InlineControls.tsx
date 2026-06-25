"use client";

import { cn } from "@/lib/utils";
import { Pencil, Trash2, Plus } from "lucide-react";
import Link from "next/link";

/** Inline edit/delete buttons shown when user is authenticated */
export function InlineEditButton({
  href,
  label = "编辑",
}: {
  href: string;
  label?: string;
}) {
  return (
    <Link
      href={href}
      className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs text-text-tertiary hover:text-accent hover:bg-accent-light/30 transition-all"
      title={label}
    >
      <Pencil size={12} />
      <span>{label}</span>
    </Link>
  );
}

export function InlineDeleteButton({
  onDelete,
  label = "删除",
}: {
  onDelete: () => void;
  label?: string;
}) {
  return (
    <button
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        if (confirm("确定删除？")) onDelete();
      }}
      className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs text-text-tertiary hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-500/10 transition-all"
      title={label}
    >
      <Trash2 size={12} />
      <span>{label}</span>
    </button>
  );
}

export function InlineCreateButton({
  href,
  label = "新建",
}: {
  href: string;
  label?: string;
}) {
  return (
    <Link
      href={href}
      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-accent text-white text-xs font-medium hover:bg-accent/90 active:scale-[0.98] transition-all"
    >
      <Plus size={13} />
      {label}
    </Link>
  );
}

/** Horizontal admin action bar with gap */
export function AdminActions({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex items-center gap-1", className)}>{children}</div>
  );
}
