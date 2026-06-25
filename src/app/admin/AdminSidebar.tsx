"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { LogoutButton } from "./LogoutButton";
import {
  LayoutDashboard,
  FileText,
  FolderKanban,
  Settings,
  ArrowLeft,
} from "lucide-react";

const navItems = [
  { href: "/admin", label: "控制台", icon: LayoutDashboard },
  { href: "/admin/posts", label: "文章管理", icon: FileText },
  { href: "/admin/projects", label: "项目管理", icon: FolderKanban },
  { href: "/admin/settings", label: "站点设置", icon: Settings },
];

export function AdminSidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-56 shrink-0 border-r border-border-light bg-bg-secondary/50 flex flex-col">
      {/* Brand */}
      <div className="px-5 py-6 border-b border-border-light">
        <Link
          href="/admin"
          className="font-display text-lg tracking-tight text-text-primary hover:text-accent transition-colors"
        >
          CMS 管理
        </Link>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-0.5">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive =
            item.href === "/admin"
              ? pathname === "/admin"
              : pathname.startsWith(item.href);

          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-all duration-200",
                isActive
                  ? "bg-accent-light dark:bg-accent-light/20 text-accent font-medium"
                  : "text-text-secondary hover:text-text-primary hover:bg-black/3 dark:hover:bg-white/5",
              )}
            >
              <Icon size={17} />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-3 py-4 border-t border-border-light space-y-0.5">
        <Link
          href="/"
          className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm text-text-secondary hover:text-text-primary hover:bg-black/3 dark:hover:bg-white/5 transition-all duration-200"
        >
          <ArrowLeft size={17} />
          <span>返回前台</span>
        </Link>
        <LogoutButton />
      </div>
    </aside>
  );
}
