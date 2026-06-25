"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { AdminLoginModal } from "@/components/admin/AdminLoginModal";
import { LogOut } from "lucide-react";

/**
 * Footer admin entry — "·" when not logged in → login modal.
 * When logged in, shows "退出" link.
 */
export function AdminEntry() {
  const [open, setOpen] = useState(false);
  const [authed, setAuthed] = useState(false);
  const router = useRouter();

  useEffect(() => {
    fetch("/api/auth/check")
      .then((r) => r.json())
      .then((d) => setAuthed(d.authenticated))
      .catch(() => {});
  }, []);

  async function handleLogout() {
    const csrf = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]*)/)?.[1] || "";
    await fetch("/api/auth/logout", {
      method: "POST",
      headers: { "x-csrf-token": csrf },
    });
    setAuthed(false);
    router.refresh();
  }

  if (authed) {
    return (
      <button
        onClick={handleLogout}
        className="inline-flex items-center gap-1 text-xs text-text-tertiary hover:text-red-500 transition-colors tracking-wider"
        title="退出登录"
      >
        <LogOut size={11} />
        退出
      </button>
    );
  }

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="text-xs text-text-tertiary hover:text-accent transition-colors tracking-wider"
        title="管理后台"
      >
        ·
      </button>
      <AdminLoginModal open={open} onClose={() => setOpen(false)} />
    </>
  );
}
