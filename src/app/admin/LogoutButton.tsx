"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { LogOut } from "lucide-react";

export function LogoutButton() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);

  async function handleLogout() {
    setLoading(true);
    await fetch("/api/auth/logout", { method: "POST" });
    router.push("/admin/login");
  }

  return (
    <button
      onClick={handleLogout}
      disabled={loading}
      className="flex items-center gap-3 w-full px-3 py-2.5 rounded-xl text-sm text-text-secondary hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-500/10 transition-all duration-200 disabled:opacity-50"
    >
      <LogOut size={17} />
      <span>{loading ? "退出中..." : "退出登录"}</span>
    </button>
  );
}
